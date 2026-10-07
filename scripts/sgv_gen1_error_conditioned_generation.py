#!/usr/bin/env python3
"""SGV-GEN1: once a real OCR error's location is known, can a model write the exact repair?

SGV-XC1 closed with outcome D. Better site discovery did not become better repair, handing every
arm the true error location added at most 0.0309 opportunity, the atomic edit representation can
express every labelable repair, and the stronger correctors it tried still rarely produced the
right replacement text. What is left is the replacement itself:

    P(exact correct replacement | real error location known)

and whether it is limited by model capacity, by the textual evidence around the site, or by the
absence of the image. This stage manipulates MODEL x EVIDENCE with the location held fixed. It is a
mechanism study, it stops at the candidate, and nothing in it is deployable.

**1. Every generation arm is oracle-localized and analysis-only.** The known site comes from the
alignment, so every arm and every artifact built from one carries `analysis_only = true,
oracle_localization = true, deployable = false`. No result here selects, ranks or deploys anything.

**2. The model sees WHERE, never WHAT.** A request carries the OCR text and a neutral marker. The
marker is `<ERR>...</ERR>` around the OCR tokens the alignment says are wrong, or `<ERR_GAP/>`
between two OCR tokens where text is missing. The gap form is geometry and it is disclosed: in this
population every gap site is an omission, so the gap marker reveals omission and nothing else, and a
span marker leaves substitution, spurious insertion and segmentation indistinguishable. No request
carries the ground-truth text, its length, its token count, the error type or any outcome.

**3. The atomic representation is SGV-XC1's, unchanged.** A candidate is a replacement for the
known site's own region, so it enters SGV-XC1's `_neural_edits` and SGV14's `_labels` exactly as
SGV-XC1's oracle-localized arms did. Nothing here re-implements a diff, a label or a metric.

**4. Evidence is a pre-registered ladder.** E0 is the site with 40 OCR characters each side, E1 the
OCR line, E2 the line with its neighbouring lines, and E3 the line plus a deterministic image crop.
The strong text model runs E0, E1 and E2, and so does the small XC1 checkpoint, so capacity and
context are crossed rather than confounded. The image arm runs E1 and E3 on the same weights, so the
visual effect is measured inside one model.

**5. Top-1 exact repair is the endpoint.** A model returns at most five ranked replacements in one
greedy decode. ExactGeneration@1 is primary; K=3 and K=5 are prefixes of the same list and are
labelled ranking headroom. Near misses, damage and span escape are diagnostics and never enter it.

**6. The design is frozen before any label exists.** Models, prompts, decoding, contexts, crops,
the site sample, the margins and the outcome rules are written by phases that refuse to run once a
label exists. A model that could not be run is recorded as unavailable before generation.

    --reconstruct   section 0: SGV-XC1's frozen state re-read from its artifacts and hashed
    --freeze        the upstream configuration and environment inventory this stage consumes
    --preregister   arms, evidence, K, margins, criteria, questions and the outcome rules
    --probe         one model's checkpoint probe (GEN1_MODEL=<arm>), while it is on disk
    --audit         one model's GT-blind format and throughput audit on adaptation documents
    --availability  the model availability audit, assembled from the probes and audits
    --sites         the known-error population, its eligibility accounting and the frozen sample
    --contexts      E0/E1/E2 text, frozen crops and the three control variants
    --registry      model, prompt, decoding, context and crop registries, frozen together
    --generate      oracle-located, GT-blind inference into an immutable per-shard cache
    --spotcheck     each model regenerates its first batches from the frozen inputs
    --normalize     parse -> SGV-XC1 atomic edits -> SGV14 labels -> one table of site outcomes
    --reproduce     section 42: SGV-XC1's known-site quantities, reproduced exactly
    --exact         ExactGeneration@K for every cell, with explicit denominators
    --strata        error type and its subtypes, engine, domain and typography
    --gains         the capacity, context and visual effects, judged by the frozen rule
    --diagnostics   near misses, wrong-edit damage and span escape
    --overlap       complementarity, unique repairs and the union ceiling
    --cost          measured runtime, tokens and memory
    --controls      the known-site controls and the three model controls
    --negative      the thirteen falsification tests
    --stats         the primary family and its secondaries, Holm-corrected
    --figures       the twenty required figures
    --decide        the machine-readable research decision
    --determinism   two regenerations from the raw cache, the inputs and the models
    --record        provenance and the traceability index

DEVELOPMENT / MECHANISM ONLY. `ready_for_external_confirmation` is false by construction: every arm
uses oracle localization on exposed development environments. The SGV1 CORD confirmatory reserve
stays LOCKED and is absent from every artifact. SGV13, SGV14, SGV15, SGV15b, SGV-DT1, SGV-CG1 and
SGV-XC1 artifacts are read and hash-verified; none is modified.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv_xc1_cross_correction_opportunity as xc1
from ocr_risk.io.hashing import canonical_hash, file_sha256

s14 = xc1.s14
s15 = xc1.s15
s15b = xc1.s15b
cg1 = xc1.cg1
dt1 = xc1.dt1

REPO = xc1.REPO
OUT = REPO / "results/generated/sgv_gen1_error_conditioned_generation"
CACHE = OUT / "cache"
# Every GEN1 input is located by ground truth, so the raw outputs live in one directory of their
# own and no GT-blind phase of any stage reads it.
RAW_CACHE = OUT / "raw_generation_outputs"
PROBE_DIR = CACHE / "probes"
AUDIT_DIR = CACHE / "format_audit"
REGENERATION_DIR = CACHE / "regeneration"
CROP_DIR = CACHE / "crops"

RESEARCH_FREEZE = OUT / "research_freeze.json"
FROZEN_CONFIGURATION = OUT / "frozen_upstream_configuration.json"
ENVIRONMENT_INVENTORY = OUT / "environment_inventory.json"
DESIGN_RECORD = OUT / "design_record.json"
ERROR_SITE_INVENTORY = OUT / "error_site_inventory.json"
MODEL_AVAILABILITY = OUT / "model_availability_audit.json"
PROMPT_FORMAT_AUDIT = OUT / "prompt_format_audit.json"
MODEL_REGISTRY = OUT / "model_registry.json"
MODEL_VERSIONS = OUT / "model_versions.json"
CONTEXT_REGISTRY = OUT / "context_registry.json"
PROMPT_REGISTRY = OUT / "prompt_registry.json"
DECODING_REGISTRY = OUT / "decoding_registry.json"
CROP_REGISTRY = OUT / "crop_registry.json"

# The site sample. The GT-blind table is what inference reads; the truth table is analysis-only
# and is never opened by a phase that builds a model input.
SAMPLE_SITES = OUT / "known_site_sample.parquet"
SAMPLE_TRUTH = OUT / "known_site_truth.parquet"
CONTEXTS = OUT / "known_site_contexts.parquet"
CONTROL_CONTEXTS = OUT / "control_contexts.parquet"

REPORT = REPO / "docs/sgv_gen1/error_conditioned_generation.md"

SCHEMA_VERSION = 1
STAGE = "sgv_gen1_error_conditioned_generation"
HYPOTHESIS = "SGV-GEN1-D1"

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
_ratio = xc1._ratio
_slug = xc1._slug
_require = xc1._require
_forbid = xc1._forbid
_digest = xc1._digest
_resolve_device = xc1._resolve_device
_release_device_cache = xc1._release_device_cache
_parameter_count = xc1._parameter_count
_checkpoint_files = xc1._checkpoint_files

BOOTSTRAP_RESAMPLES = xc1.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = xc1.BOOTSTRAP_SEED
ALPHA = xc1.ALPHA
ENVIRONMENT_COUNT = xc1.ENVIRONMENT_COUNT
CONTEXT_CHARS = xc1.CONTEXT_CHARS
OUTCOME_EXACT = xc1.OUTCOME_EXACT
REQUEST_COLUMNS = xc1.REQUEST_COLUMNS
LABEL_SITE_COLUMNS = xc1.LABEL_SITE_COLUMNS
RAW_COLUMNS = xc1.RAW_COLUMNS
XC1_ORACLE = xc1.COND_ORACLE_LOC
C3_ORACLE = xc1.C3_ORACLE
LOCAL_MODEL_ROOT = xc1.LOCAL_MODEL_ROOT

UPSTREAM_SCRIPTS = (
    *xc1.UPSTREAM_SCRIPTS,
    "scripts/sgv_xc1_cross_correction_opportunity.py",
)

# ------------------------------------------------------------------ section 0: the frozen XC1 state
#
# Section 0 lists what the brief believes SGV-XC1 found. Each is confirmed from SGV-XC1's own
# artifacts at the brief's four-decimal precision; a disagreement stops the stage.

XC1_STATE = ("COMPLETE", "D", "NEXT: SITE-LOCALIZATION / REPRESENTATION INVESTIGATION")
XC1_EXPECTED = {
    "m0_opportunity": 0.0960,
    "m1_opportunity": 0.0096,
    "m2_opportunity": 0.0178,
    "union_opportunity": 0.1192,
    "oracle_localization_ceiling": 0.0974,
}
EXPECTATION_TOLERANCE = 5e-5
# "Omissions remain essentially unrepaired": every arm's omission opportunity stays below this.
OMISSION_UNREPAIRED_BELOW = 0.01

# ------------------------------------------------------------------ the pre-registered design
#
# Everything from here to the end of this block is written to `design_record.json` and the
# registries before a single label exists, by phases that refuse to run afterwards.

M0 = xc1.M0
M1 = xc1.M1
M2 = xc1.M2
M2G = "m2g_qwen25_1p5b_gen1"
M3 = "m3_qwen3_4b_instruct"
M4 = "m4_phi4_mini_instruct"
M5 = "m5_qwen3_vl_4b_instruct"
M_UNION = "m_union_gen1"

# Section 8. The frozen XC1 baselines are read from SGV-XC1's oracle-localization tables and keep
# their own contracts; they are the reproduction gate and the M0 comparison, not GEN1 arms.
XC1_BASELINES = (M0, M1, M2)
GEN1_MODELS = (M2G, M3, M4, M5)
STRONG_MODELS = (M3, M4, M5)

MODEL_SPECS: dict[str, dict[str, Any]] = {
    M2G: {
        "repo_id": xc1.LLM_REPO,
        "revision": xc1.LLM_REVISION,
        "kind": "text",
        "family": "Qwen2.5",
        "developer": "Qwen team, Alibaba Cloud",
        "role": "small-capacity reference: SGV-XC1's M2 checkpoint re-run under the GEN1 contract",
        "reason_for_inclusion": (
            "section 28 compares capacity at fixed evidence, which requires the XC1 checkpoint "
            "under exactly the prompt, context and decoding the strong models receive; SGV-XC1's "
            "own M2 outputs used a different prompt and one candidate."
        ),
        "batch_size": 32,
    },
    M3: {
        "repo_id": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "kind": "text",
        "family": "Qwen3",
        "developer": "Qwen team, Alibaba Cloud",
        "role": "strong contemporary multilingual text model",
        "reason_for_inclusion": (
            "a later model generation than M2 rather than a larger copy of it (section 8): "
            "released 2025-08 against M2's 2024-09, instruction-tuned without a thinking mode so a "
            "greedy answer is the answer, and the largest current open-weight text model whose "
            "bf16 weights fit this machine's 16 GiB of unified memory beside the OS."
        ),
        "batch_size": 16,
    },
    M4: {
        "repo_id": "microsoft/Phi-4-mini-instruct",
        "revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
        "kind": "text",
        "family": "Phi-4",
        "developer": "Microsoft",
        "role": "strong text model from a distinct family",
        "reason_for_inclusion": (
            "section 8 M4: a second developer and training lineage, so no conclusion rests on "
            "one family. Its card declares German among its languages; it is loaded through "
            "transformers' native Phi-3 implementation, never the checkpoint's own code."
        ),
        "batch_size": 16,
    },
    M5: {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "kind": "vision",
        "family": "Qwen3-VL",
        "developer": "Qwen team, Alibaba Cloud",
        "role": "image-aware model; its text-only run is the matched comparator",
        "reason_for_inclusion": (
            "section 29 forbids crediting vision when the image model is also larger: running the "
            "same weights with and without the crop makes the visual effect a within-model "
            "contrast. It is the open-weight image-text model of this generation that fits this "
            "machine; gated checkpoints (Gemma 3, Llama 3.2 Vision) are unavailable without an "
            "access token, which this environment does not have."
        ),
        "batch_size": 8,
    },
}

# Section 10. The evidence ladder, and the three control variants of sections 46-48.
E0 = "e0_local"
E1 = "e1_line"
E2 = "e2_neighbour_lines"
E3 = "e3_line_image"
X_MARK = "x_marker_shuffled"
X_CTX = "x_context_shuffled"
X_IMG = "x_image_shuffled"
EVIDENCE = (E0, E1, E2, E3)
CONTROL_CONDITIONS = (X_MARK, X_CTX, X_IMG)

PRIMARY_CELLS: tuple[tuple[str, str], ...] = (
    (M2G, E0),
    (M2G, E1),
    (M2G, E2),
    (M3, E0),
    (M3, E1),
    (M3, E2),
    (M4, E1),
    (M5, E1),
    (M5, E3),
)
# Each control runs on the control subsample and is read against the cell named beside it.
CONTROL_CELLS: tuple[tuple[str, str, str], ...] = (
    (M3, X_MARK, E2),
    (M3, X_CTX, E2),
    (M5, X_IMG, E3),
)
CELL_CONDITIONS = {model: [c for m, c in PRIMARY_CELLS if m == model] for model in GEN1_MODELS}
CONTROL_CONDITIONS_BY_MODEL = {
    model: [c for m, c, _ref in CONTROL_CELLS if m == model] for model in GEN1_MODELS
}

# Section 14. One greedy decode returns a ranked list; every K is a prefix of it.
MAX_CANDIDATES = 5
K_GRID = (1, 3, 5)
PRIMARY_K = 1

# Sections 52-54. Margins, fixed here and never moved after a result is read.
MEANINGFUL_DELTA = 0.10
WEAK_TYPE_DELTA = 0.10
WEAK_ERROR_TYPES = xc1.WEAK_ERROR_TYPES
WEAK_TYPES_REQUIRED = 2
STRONG_ERROR_TYPE = xc1.STRONG_ERROR_TYPE
BREADTH_MAJORITY = 6

# The site sample. Four models over three evidence levels do not fit this machine at the full
# 28,126-site population, so a fixed number of sites per environment is drawn content-blind, by a
# hash of the site id, and every arm is scored on the same sites. The number is a constant, fixed
# from the GT-blind M2G audit's measured throughput before any evaluation output existed. No label
# informs it.
GEN1_SEED = 20260913
SAMPLE_PER_ENVIRONMENT = 150
CONTROL_PER_ENVIRONMENT = 60
SUPERSEDED_DIR = CACHE / "superseded"
SUPERSEDED_DESIGN = SUPERSEDED_DIR / "design_record.v0.json"
SUPERSEDED_DESIGN_V1 = SUPERSEDED_DIR / "design_record.v1.json"
SUPERSEDED_AUDIT = SUPERSEDED_DIR / "format_audit.m2g_qwen25_1p5b_gen1.decoding_rev0.json"
REISSUE_REASONS = (
    "reissued before any site was sampled and before any model was audited or generated, to "
    "replace the throughput-ladder sample rule with a constant and to sequence the registries per "
    "model, because the download rate made auditing every checkpoint before any generation "
    "impractical. No arm, prompt, decoding setting, margin, criterion or outcome rule changed.",
    "reissued after the GT-blind M2G audit on adaptation documents and before any evaluation "
    "output or label existed, to reduce the sample to 150 sites per environment and the control "
    "subsample to 60: under the five-candidate contract the audit measured a fraction of SGV-XC1's "
    "M2 throughput, at which 250 sites would not fit an overnight run. The new sample is the first "
    "150 sites of the same content-blind order. No arm, prompt, margin, criterion or outcome rule "
    "changed; the output cap was revised in the decoding registry on the same audit.",
)

OUTCOME_TAXONOMY = {
    "A": "model capacity is load-bearing",
    "B": "context / evidence is load-bearing",
    "C": "capacity and evidence are complementary",
    "D": "exact generation remains fundamentally weak",
}
QUESTIONS = {
    "Q1": (
        "Does stronger model capacity materially improve exact replacement generation once "
        "localization is solved?"
    ),
    "Q2": "Does wider textual context materially improve exact generation?",
    "Q3": "Does image evidence materially improve generation beyond matched text context?",
    "Q4": "Which evidence source helps substitutions, omissions, and segmentation?",
    "Q5": (
        "Is omission failure primarily a lack-of-context problem or a generation-capacity problem?"
    ),
    "Q6": (
        "Does any tested condition raise exact generation enough to justify building a new "
        "candidate universe?"
    ),
}

# Section 50. The primary family: four comparisons, Holm-corrected together.
PRIMARY_COMPARISONS: tuple[tuple[str, tuple[str, str], tuple[str, str]], ...] = (
    ("P1_capacity", (M3, E1), (M2G, E1)),
    ("P2_extended_context", (M3, E2), (M3, E1)),
    ("P3_visual", (M5, E3), (M5, E1)),
)
P4_NAME = "P4_strongest_arm_vs_m0"

# The factor effects of sections 27-29. Each is "material" only if it clears the margin on the
# mean over environments, in a majority of environments, and in two of the three weak types.
CAPACITY_EFFECTS: tuple[tuple[str, tuple[str, str], tuple[str, str]], ...] = (
    ("capacity_m3_e0", (M3, E0), (M2G, E0)),
    ("capacity_m3_e1", (M3, E1), (M2G, E1)),
    ("capacity_m3_e2", (M3, E2), (M2G, E2)),
    ("capacity_m4_e1", (M4, E1), (M2G, E1)),
    ("capacity_m5_text_e1", (M5, E1), (M2G, E1)),
)
EVIDENCE_EFFECTS: tuple[tuple[str, tuple[str, str], tuple[str, str]], ...] = (
    ("context_m3_e1_vs_e0", (M3, E1), (M3, E0)),
    ("context_m3_e2_vs_e1", (M3, E2), (M3, E1)),
    ("context_m3_e2_vs_e0", (M3, E2), (M3, E0)),
    ("context_m2g_e1_vs_e0", (M2G, E1), (M2G, E0)),
    ("context_m2g_e2_vs_e1", (M2G, E2), (M2G, E1)),
    ("context_m2g_e2_vs_e0", (M2G, E2), (M2G, E0)),
    ("visual_m5_e3_vs_e1", (M5, E3), (M5, E1)),
)
JOINT_EFFECTS: tuple[tuple[str, tuple[str, str], tuple[str, str]], ...] = (
    ("joint_m3_e2_vs_m2g_e0", (M3, E2), (M2G, E0)),
    ("joint_m5_e3_vs_m2g_e1", (M5, E3), (M2G, E1)),
)
OUTCOME_RULE = {
    "material": (
        f"an effect is material when its mean over environments is >= {MEANINGFUL_DELTA}, it is "
        f">= {MEANINGFUL_DELTA} in >= {BREADTH_MAJORITY} of {ENVIRONMENT_COUNT} environments, and "
        f">= {WEAK_TYPES_REQUIRED} of {list(WEAK_ERROR_TYPES)} improve by >= {WEAK_TYPE_DELTA} "
        "pooled -- all at K=1"
    ),
    "A": "some capacity effect is material and no evidence effect is",
    "B": "some evidence effect is material and no capacity effect is",
    "C": (
        "a joint effect is material and either both a capacity and an evidence effect are "
        "material, or neither is -- only the combination clears the margin"
    ),
    "D": "no capacity, evidence or joint effect is material",
    "note": (
        "section 53: an effect carried only by spurious insertion cannot be material, because "
        "the weak-type clause is required. Section 55: K=3 and K=5 never enter the rule."
    ),
}

# ------------------------------------------------------------------ section 12: the output contract

ERR_OPEN = "<ERR>"
ERR_CLOSE = "</ERR>"
ERR_GAP = "<ERR_GAP/>"
PROMPT_REVISION = 0
SYSTEM_PROMPT = (
    "You correct OCR errors. The user shows text produced by an OCR engine in which one position "
    "is marked as a real OCR error:\n"
    f"- {ERR_OPEN}...{ERR_CLOSE} encloses OCR text that was recognised wrongly.\n"
    f"- {ERR_GAP} marks a place where the OCR engine missed text.\n"
    "Propose the correct text for the marked position only.\n\n"
    "Answer with a single JSON object and nothing else, in this form:\n"
    '{"candidates": ["first choice", "second choice"]}\n'
    "Rules:\n"
    f"1. Give at most {MAX_CANDIDATES} different candidates, the most likely first.\n"
    f"2. For {ERR_OPEN}...{ERR_CLOSE}, each candidate is the complete replacement for the "
    'enclosed text. Use "" if the enclosed text should be removed.\n'
    f"3. For {ERR_GAP}, each candidate is the missing text to insert at that position.\n"
    "4. Never include text from outside the marked position, and never include the markers.\n"
    "5. Keep the document's own spelling, capitalisation, punctuation and letter forms, "
    "including historical ones. Do not modernise.\n"
    '6. If you cannot propose a correction, answer {"candidates": []}.'
)
USER_TEMPLATES = {
    E0: "OCR text around the error:\n{text}",
    E1: "OCR line:\n{text}",
    E2: "OCR text (the line with the error and its neighbouring lines):\n{text}",
    E3: "The image shows the document around the error.\nOCR line:\n{text}",
}
# A control condition uses the template of the evidence level whose layout it copies.
TEMPLATE_OF = {E0: E0, E1: E1, E2: E2, E3: E3, X_MARK: E2, X_CTX: E2, X_IMG: E3}
PROMPT_DRAFTS: list[dict[str, Any]] = []

# Section 15. Greedy, one sequence, repetition penalty neutralized so every model decodes by the
# same rule whatever its checkpoint's generation_config suggests for sampling.
# Revision 1, made on the GT-blind M2G audit before any evaluation output existed: the first rule
# budgeted a token per region character and let a runaway decode hold its batch open to the cap,
# while the answers that finished used a small part of it. Tokens are now budgeted at two
# characters each, and a list the cap cuts off keeps the candidates it completed.
DECODING_REVISION = 1
GEN_MAX_NEW_TOKENS = 160
GEN_BASE_TOKENS = 24
GEN_PER_CANDIDATE_MARGIN = 6
GAP_ALLOWANCE_CHARS = 16
CHARS_PER_TOKEN = 2
MODEL_DTYPE = "bfloat16"
MAX_NEW_TOKENS_RULE = (
    f"min({GEN_MAX_NEW_TOKENS}, {GEN_BASE_TOKENS} + {MAX_CANDIDATES} x (ceil(widest region in "
    f"the batch in characters, or {GAP_ALLOWANCE_CHARS} at a gap, / {CHARS_PER_TOKEN}) + "
    f"{GEN_PER_CANDIDATE_MARGIN}))"
)
DECODING_HISTORY = (
    {
        "revision": 0,
        "status": "superseded before any evaluation output existed",
        "rule": (
            "min(256, 32 + 5 x (widest region in the batch in characters, or 24 at a gap, + 8))"
        ),
        "cut_off_list": "the whole answer discarded",
        "reason": (
            "the GT-blind M2G audit on adaptation documents found answers running to the cap in "
            "every condition, each holding its batch open, at a fraction of SGV-XC1's M2 "
            "throughput; a cut-off list also lost the candidates it had completed. Revised once "
            "and frozen afterwards whatever later audits show. No label was read."
        ),
    },
)

# Section 37. The crop, frozen before any outcome.
CROP_HEIGHT = 64
CROP_MAX_WIDTH = 1024
CROP_PAD_FRACTION = 0.25
CROP_MIN_PAD = 4

# The GT-blind format audit: adaptation documents only, SGV-XC1's two audit environments.
AUDIT_ENVIRONMENTS = xc1.AUDIT_ENVIRONMENTS
AUDIT_REGIONS = 32
AUDIT_GAPS = 16

# Columns that may never appear in a table a model input is built from (section 49).
FORBIDDEN_INPUT_COLUMNS = (
    "region_gt",
    "gt_text",
    "gt_length",
    "gt_tokens",
    "site_kind",
    "true_error_type",
    "outcome",
    "exact",
    "beneficial",
    "is_harmful",
    "harmful",
    "d_after",
    "align_site_id",
)


# ------------------------------------------------------------------ small shared helpers


def _envelope(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "schema_version": f"sgv_gen1-{artifact}-v{SCHEMA_VERSION}",
        "stage": STAGE,
        "hypothesis_id": HYPOTHESIS,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _analysis_envelope(artifact: str) -> dict[str, Any]:
    """Section 4. Every artifact built from an oracle-located arm declares it in its header."""
    return {
        **_envelope(artifact),
        "analysis_only": True,
        "oracle_localization": True,
        "deployable": False,
        "uses_ground_truth": True,
        "may_select_a_deployable_method": False,
        "may_influence_a_prompt_or_decoding_setting": False,
    }


def environment_specs() -> list[dict[str, Any]]:
    return list(s14.ENVIRONMENTS)


def _spec(name: str) -> dict[str, Any]:
    return next(spec for spec in s14.ENVIRONMENTS if spec["environment"] == name)


def _xc1_json(name: str) -> dict[str, Any]:
    return cc_read_json(xc1.OUT / f"{name}.json")


def _labels_exist() -> bool:
    return (OUT / "generation_candidates.parquet").exists()


def _model_dir(model: str) -> Path:
    spec = MODEL_SPECS[model]
    return LOCAL_MODEL_ROOT / f"{spec['repo_id'].split('/')[-1]}@{spec['revision'][:12]}"


def _snapshot(model: str) -> Path | None:
    path = _model_dir(model)
    return path if (path / "config.json").is_file() else None


def _stable_seed(*parts: str) -> int:
    return int(canonical_hash({"seed": GEN1_SEED, "parts": list(parts)})[:16], 16)


def _order_key(site_id: str) -> str:
    """The content-blind sampling order: a hash of the site id and the stage seed, nothing else."""
    return canonical_hash({"seed": GEN1_SEED, "site_id": site_id})


# ------------------------------------------------------------------ the traceability auditor
#
# Ported from the SGV-XC1 audit so both reports are held to one rule: every number is recoverable
# from an artifact its own section names. It is applied to SGV-XC1's report in section 0 and to
# this stage's report in --record.

_DECIMAL = re.compile(r"-?\d+\.\d+")
_GROUPED = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_PERCENT = re.compile(r"-?\d+(?:\.\d+)?(?=%)")
_INTEGER = re.compile(r"(?<![\w.,\-§])\d+(?![\w.,%])")
_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_CODE_SPAN = re.compile(r"`[^`]*`")
_CROSS_REFERENCE = re.compile(
    r"(?:[Ss]ections?|[Cc]ontrols?|[Cc]riteri(?:on|a)|[Qq]uestions?|[Ff]alsification tests?"
    r"|SGV-?(?:XC1|XR1|CG1|DT1|GEN1)|SGV15b?|SGV1[0-9]?|[Ff]igure|[Tt]able|[NCQKHPEM])\s*\d+"
)


def _walk_numbers(node: object, sink: set[float]) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        value = float(node)
        if value == value and abs(value) != float("inf"):
            sink.add(value)
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _walk_numbers(key, sink)
            _walk_numbers(value, sink)
        sink.add(float(len(node)))
        return
    if isinstance(node, list):
        for value in node:
            _walk_numbers(value, sink)
        sink.add(float(len(node)))
        return
    if isinstance(node, str):
        for pattern in (_DECIMAL, _GROUPED, _PERCENT):
            for match in pattern.findall(node):
                with contextlib.suppress(ValueError):
                    sink.add(float(match.replace(",", "")))


def _number_pool(paths: Sequence[Path]) -> set[float]:
    sink: set[float] = set()
    for path in paths:
        if path.is_file() and path.suffix == ".json":
            _walk_numbers(json.loads(path.read_text(encoding="utf-8")), sink)
    derived = {v * 100.0 for v in sink} | {v / 100.0 for v in sink} | {-v for v in sink}
    return sink | derived | {abs(v) for v in sink}


def _report_sections(text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    current: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current is not None:
                parts.append((current, "\n".join(buffer)))
            current, buffer = line[3:].strip(), []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        parts.append((current, "\n".join(buffer)))
    return parts


def _strip_structure(body: str) -> str:
    """Code, cross-references and markdown structure are not numeric claims about results."""
    body = _CROSS_REFERENCE.sub(" ", _CODE_SPAN.sub(" ", _CODE_BLOCK.sub(" ", body)))
    lines = []
    for line in body.splitlines():
        if line.strip().startswith("|") and set(line.strip()) <= set("|-: "):
            continue
        lines.append(re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line))
    return "\n".join(lines)


def audit_report(report: Path, sections: dict[str, Sequence[Path]]) -> dict[str, Any]:
    """Every decimal, grouped integer, percentage and integer, traced to its section's artifacts.

    A value is traced if one of the section's artifacts holds it exactly, or after rounding to the
    digits written, or as a percentage of a stored fraction, or as the size of a stored list.
    """
    counted = {"decimal": 0, "grouped": 0, "percent": 0, "integer": 0}
    untraceable: list[str] = []
    for name, body in _report_sections(report.read_text(encoding="utf-8")):
        paths = sections.get(name)
        if paths is None:
            untraceable.append(f"{name}: section not indexed")
            continue
        pool = _number_pool(paths)
        text = _strip_structure(body)
        for kind, pattern in (
            ("decimal", _DECIMAL),
            ("grouped", _GROUPED),
            ("percent", _PERCENT),
            ("integer", _INTEGER),
        ):
            for match in pattern.findall(text):
                literal = match.replace(",", "")
                value = float(literal)
                digits = len(literal.split(".")[1]) if "." in literal else 0
                tolerance = max(0.5 * 10.0 ** (-digits) if digits else 1e-9, 1e-12)
                counted[kind] += 1
                if not any(abs(value - candidate) <= tolerance for candidate in pool):
                    untraceable.append(f"{name}: {kind} {match}")
    return {
        "numeric_claims": sum(counted.values()),
        "by_class": counted,
        "untraceable_numeric_claims": len(untraceable),
        "untraceable": untraceable,
    }


# ------------------------------------------------------------------ section 0: reconstruct


def _xc1_checks() -> tuple[dict[str, Any], list[str]]:
    """Every section-0 statement, read from SGV-XC1's artifacts: the record and any mismatch."""
    decision = _xc1_json("research_decision")
    opportunity = _xc1_json("opportunity_results")
    by_method = {
        str(row["method"]): row
        for row in opportunity["conditions"][xc1.PRIMARY_CONDITION]["by_method"]
    }
    oracle = _xc1_json("oracle_localization")
    union = _xc1_json("union_opportunity")
    error_types = _xc1_json("error_type_analysis")
    reproduction = _xc1_json("upstream_reproduction")
    determinism = _xc1_json("determinism")
    provenance = _xc1_json("provenance")
    mismatches: list[str] = []

    state = (
        str(decision["status"]),
        str(decision["outcome"]),
        str(decision["recommended_next_stage"]),
    )
    if state != XC1_STATE:
        mismatches.append(f"sgv_xc1 state {state!r} != {XC1_STATE!r}")
    for flag in ("ready_for_external_confirmation", "confirmatory_reserve_consumed"):
        if decision[flag] is not False:
            mismatches.append(f"sgv_xc1 {flag} is not false")

    decomposition = decision["oracle_localization"]
    observed = {
        "m0_opportunity": float(decision["baseline_opportunity"]),
        "m1_opportunity": float(by_method[M1]["mean_opportunity_recall"]),
        "m2_opportunity": float(by_method[M2]["mean_opportunity_recall"]),
        "union_opportunity": float(union["union_all_methods"]["mean_opportunity_recall"]),
        "oracle_localization_ceiling": max(
            float(row["mean_oracle_localization"]) for row in decomposition.values()
        ),
    }
    for key, expected in XC1_EXPECTED.items():
        if abs(observed[key] - expected) > EXPECTATION_TOLERANCE:
            mismatches.append(f"sgv_xc1 {key}: observed {observed[key]:.6f} != {expected}")

    discovery_delta = float(decision["discovery_delta"])
    generation_delta = float(decision["generation_delta"])
    if not (discovery_delta > 0 > generation_delta):
        mismatches.append("sgv_xc1: M2 discovery is not above M0 with generation below it")
    headroom = {m: float(r["mean_localization_headroom"]) for m, r in decomposition.items()}
    if max(headroom.values()) >= MEANINGFUL_DELTA:
        mismatches.append(f"sgv_xc1: oracle localization rescues an arm: {headroom}")

    c3 = oracle["c3_ground_truth_oracle_corrector"]
    if int(c3["repaired_error_sites"]) != int(c3["expressible_labelable_error_sites"]):
        mismatches.append("sgv_xc1: the GT corrector does not express every labelable repair")

    rows = error_types["by_condition"][xc1.PRIMARY_CONDITION]
    by_type = {
        method: {str(r["stratum"]): float(r["opportunity_recall"]) for r in rows[method]}
        for method in rows
    }
    recovery = error_types["weak_type_recovery"]
    deltas = {m: {k: float(v) for k, v in r["delta_by_type"].items()} for m, r in recovery.items()}
    if not all(deltas[m]["substitution"] > 0 for m in deltas):
        mismatches.append("sgv_xc1: substitutions do not improve under the neural arms")
    if not all(by_type[m]["deletion"] < OMISSION_UNREPAIRED_BELOW for m in by_type):
        mismatches.append("sgv_xc1: omissions are not essentially unrepaired")
    if not all(deltas[m]["segmentation"] < MEANINGFUL_DELTA for m in deltas):
        mismatches.append("sgv_xc1: a segmentation improvement is not small")
    strong = by_type[M0][STRONG_ERROR_TYPE]
    if not all(strong > 10 * by_type[M0][weak] for weak in WEAK_ERROR_TYPES):
        mismatches.append("sgv_xc1: M0's strength is not spurious-insertion repair")

    if not decision["baseline_matches_sgv_cg1"] or not all(
        c["agrees"] for c in reproduction["checks"]
    ):
        mismatches.append("sgv_xc1: the SGV-CG1 reproduction did not pass")
    if not determinism["all_runs_identical"]:
        mismatches.append("sgv_xc1: determinism did not pass")
    unchanged = provenance["upstream_unchanged_since_section_zero"]
    upstream_now = {
        path: file_sha256(REPO / path) == digest
        for path, digest in provenance["upstream_inputs"].items()
    }
    if (
        unchanged["sgv_cg1_artifacts_changed"]
        or unchanged["upstream_scripts_changed"]
        or not unchanged["upstream_decisions_unchanged"]
        or not all(upstream_now.values())
    ):
        mismatches.append("sgv_xc1: an input SGV-XC1 recorded as upstream has changed")

    trace = _xc1_json("traceability")
    audit = audit_report(
        xc1.REPORT, {name: [REPO / p for p in paths] for name, paths in trace["sections"].items()}
    )
    if audit["untraceable_numeric_claims"]:
        mismatches.append(f"sgv_xc1 report: {audit['untraceable_numeric_claims']} untraceable")

    record = {
        "state": dict(zip(("status", "outcome", "recommended_next_stage"), state, strict=True)),
        "ready_for_external_confirmation": decision["ready_for_external_confirmation"],
        "confirmatory_reserve_consumed": decision["confirmatory_reserve_consumed"],
        "headline": observed,
        "discovery_delta_m2": discovery_delta,
        "generation_delta_m2": generation_delta,
        "localization_headroom": headroom,
        "generation_headroom": {
            m: float(r["mean_generation_headroom"]) for m, r in decomposition.items()
        },
        "oracle_localization_by_method": {
            m: float(r["mean_oracle_localization"]) for m, r in decomposition.items()
        },
        "c3_ground_truth_corrector": {
            "repaired_error_sites": int(c3["repaired_error_sites"]),
            "expressible_labelable_error_sites": int(c3["expressible_labelable_error_sites"]),
            "expressible_error_sites": int(c3["expressible_error_sites"]),
            "ocr_error_sites": int(c3["ocr_error_sites"]),
        },
        "opportunity_by_error_type": by_type,
        "weak_type_delta_by_method": deltas,
        "sgv_cg1_reproduction_checks": len(reproduction["checks"]),
        "sgv_cg1_reproduction_agreeing": sum(bool(c["agrees"]) for c in reproduction["checks"]),
        "determinism_all_runs_identical": bool(determinism["all_runs_identical"]),
        "upstream_inputs_rehashed": len(upstream_now),
        "upstream_inputs_unchanged": sum(upstream_now.values()),
        "report_traceability": {
            key: audit[key] for key in ("numeric_claims", "by_class", "untraceable_numeric_claims")
        },
    }
    return record, mismatches


def run_reconstruct() -> int:
    """Section 0 and 1: SGV-XC1's frozen state, confirmed and hashed before anything else runs."""
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    _forbid(RESEARCH_FREEZE)
    record, mismatches = _xc1_checks()
    if mismatches:
        raise PhaseError(
            "section 0 reconstruction failed; SGV-GEN1 must not begin:\n  "
            + "\n  ".join(mismatches)
        )
    upstream_states = {
        "sgv15": list(
            cg1._decision_state(s15.DECISION, ("verdict", "criteria_met", "ready_for_sgv16"))
        ),
        "sgv15b": list(
            cg1._decision_state(s15b.DECISION, ("verdict", "criteria_met", "ready_for_sgv16"))
        ),
        "sgv_dt1": list(
            cg1._decision_state(
                dt1.DECISION, ("verdict", "criteria_met", "ready_for_external_confirmation")
            )
        ),
    }
    xc1_files = sorted(
        [
            *xc1.OUT.glob("*.json"),
            *xc1.OUT.glob("*.parquet"),
            *xc1.RAW_CACHE.glob("*.parquet"),
            *xc1.RAW_ORACLE_CACHE.glob("*.parquet"),
        ]
    )
    cg1_files = sorted([*cg1.OUT.glob("*.json"), *cg1.OUT.glob("*.parquet")])
    status = _git("status", "--porcelain", "--", *(str(REPO / p) for p in UPSTREAM_SCRIPTS))
    _write_json_once(
        RESEARCH_FREEZE,
        {
            **_envelope("research_freeze"),
            "head": _git("rev-parse", "HEAD"),
            "gen1_starting_commit": _git("rev-parse", "HEAD"),
            "sgv_xc1": record,
            "upstream_decisions": upstream_states,
            "sgv_xc1_artifacts": {_relative(p): file_sha256(p) for p in xc1_files},
            "sgv_cg1_artifacts": {_relative(p): file_sha256(p) for p in cg1_files},
            "upstream_scripts": {name: file_sha256(REPO / name) for name in UPSTREAM_SCRIPTS},
            "sgv_xc1_commit_state": {
                "committed": _tracked(REPO / "scripts/sgv_xc1_cross_correction_opportunity.py"),
                "upstream_script_status": status.splitlines(),
                "note": (
                    "section 1 commits SGV-XC1 only if explicitly authorized. No authorization "
                    "was given, so SGV-XC1 is frozen here by content hash: every artifact, raw "
                    "output and upstream script above is re-hashed by --record and must be "
                    "unchanged. Its documented provenance defects are left as issued."
                ),
            },
            "confirmatory_reserve_consumed": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"reconstruct: SGV-XC1 {XC1_STATE[1]} confirmed, opportunity M0 "
        f"{record['headline']['m0_opportunity']:.4f} M1 {record['headline']['m1_opportunity']:.4f} "
        f"M2 {record['headline']['m2_opportunity']:.4f} union "
        f"{record['headline']['union_opportunity']:.4f}; XC1 report "
        f"{record['report_traceability']['numeric_claims']} claims, "
        f"{record['report_traceability']['untraceable_numeric_claims']} untraceable; "
        f"{len(xc1_files)} XC1 files hashed"
    )
    return 0


# ------------------------------------------------------------------ section 0: freeze


@dataclass(frozen=True, slots=True)
class Page:
    """One (document, engine) page in OCR terms only: its stream, spans, lines and image.

    Built from canonical OCR spans and the page's image metadata. It holds no ground-truth field,
    so nothing built from it can carry one; the leakage suite checks the builder's source.
    """

    stream: str
    span_start: dict[str, int]
    span_end: dict[str, int]
    span_line: dict[str, int]
    span_box: dict[str, tuple[float, float, float, float] | None]
    order: tuple[str, ...]
    lines: tuple[tuple[int, int], ...]
    image_path: str
    image_sha256: str
    width: int
    height: int


def _page_index(spec: dict[str, Any]) -> dict[tuple[str, str], Page]:
    """Every page of one environment, from the canonical OCR spans and the image metadata."""
    from ocr_risk.canonical import rebuild_stream
    from ocr_risk.io.paths import data_root

    bundles = s14._document_bundles(spec["corpus"])
    spans, _record = s14._canonical_spans(spec, bundles)
    root = data_root()
    pages: dict[tuple[str, str], Page] = {}
    for pair, group in spans.items():
        if not group:
            continue
        stream = rebuild_stream(group)
        ordered = sorted(group, key=lambda s: (s.char_start, s.char_end, s.span_id))
        keys: dict[str, list[Any]] = {}
        for span in ordered:
            key = str(span.line_id) if span.line_id is not None else f"__span__{span.span_id}"
            keys.setdefault(key, []).append(span)
        extents = sorted(
            (min(s.char_start for s in members), max(s.char_end for s in members), key)
            for key, members in keys.items()
        )
        line_of_key = {key: index for index, (_s, _e, key) in enumerate(extents)}
        document = bundles[pair[0]].document
        pages[pair] = Page(
            stream=stream,
            span_start={s.span_id: int(s.char_start) for s in ordered},
            span_end={s.span_id: int(s.char_end) for s in ordered},
            span_line={
                s.span_id: line_of_key[
                    str(s.line_id) if s.line_id is not None else f"__span__{s.span_id}"
                ]
                for s in ordered
            },
            span_box={
                s.span_id: (
                    (float(s.bbox.x0), float(s.bbox.y0), float(s.bbox.x1), float(s.bbox.y1))
                    if s.bbox is not None
                    else None
                )
                for s in ordered
            },
            order=tuple(s.span_id for s in ordered),
            lines=tuple((int(start), int(end)) for start, end, _key in extents),
            image_path=str(root / document.image_path),
            image_sha256=str(document.image_sha256),
            width=int(document.width),
            height=int(document.height),
        )
    return pages


def run_freeze() -> int:
    """The upstream configuration held fixed, and what each environment offers an image arm."""
    started = time.monotonic()
    _require(RESEARCH_FREEZE, "reconstruct")
    for path in (FROZEN_CONFIGURATION, ENVIRONMENT_INVENTORY):
        _forbid(path)
    xc1_configuration = _xc1_json("frozen_upstream_configuration")
    oracle = _xc1_json("oracle_localization")
    expressibility = {row["environment"]: row for row in oracle["expressibility"]}
    inventory: list[dict[str, Any]] = []
    for spec in environment_specs():
        name = spec["environment"]
        bundles = s14._document_bundles(spec["corpus"])
        roles = s14.partition_of(spec["corpus"], sorted(bundles))
        evaluation = sorted(d for d, role in roles.items() if role == s14.ROLE_EVALUATION)
        pages = _page_index(spec)
        evaluated = [pages[p] for p in pages if p[0] in set(evaluation)]
        boxes = sum(sum(v is not None for v in page.span_box.values()) for page in evaluated)
        spans = sum(len(page.span_box) for page in evaluated)
        inventory.append(
            {
                "environment": name,
                "corpus": spec["corpus"],
                "domain": "modern_forms" if spec["corpus"] == "funsd" else "historical_print",
                "base_engine": spec["base_engine"],
                "engine_id": spec["engine_id"],
                "language": xc1.language_of(spec["corpus"]),
                "typography": xc1._typography(spec),
                "evaluation_documents": len(evaluation),
                "evaluation_pages_with_ocr": len(evaluated),
                "evaluation_images_present": sum(Path(p.image_path).is_file() for p in evaluated),
                "evaluation_spans": spans,
                "evaluation_spans_with_box": boxes,
                "ocr_error_sites": int(expressibility[name]["ocr_error_sites"]),
                "expressible_error_sites": int(expressibility[name]["expressible"]),
            }
        )
        print(f"  {name}: {len(evaluated)} pages, {boxes}/{spans} spans with a box", flush=True)
    _write_json_once(
        FROZEN_CONFIGURATION,
        {
            **_envelope("frozen_upstream_configuration"),
            "sgv_xc1_frozen_configuration": {
                "artifact": _relative(xc1.FROZEN_CONFIGURATION),
                "sha256": file_sha256(xc1.FROZEN_CONFIGURATION),
                "environments": xc1_configuration["environments"],
                "alignment_config": xc1_configuration["alignment_config"],
                "site_config": xc1_configuration["site_config"],
                "labelling": xc1_configuration["labelling"],
            },
            "atomic_edit_schema": {
                "artifact": _relative(xc1.ATOMIC_EDIT_SCHEMA),
                "sha256": file_sha256(xc1.ATOMIC_EDIT_SCHEMA),
                "note": (
                    "section 6: SGV-XC1's representation, unchanged. A GEN1 candidate replaces "
                    "the known site's own region, the case SGV-XC1's region conditions already "
                    "define, so no new format and no new diff is introduced."
                ),
            },
            "known_site_construction": {
                "function": "sgv_xc1_cross_correction_opportunity._oracle_requests",
                "cache": _relative(xc1.CACHE),
                "note": (
                    "the known-site requests are SGV-XC1's oracle-localization requests, read "
                    "from its cache: the OCR region of each error site located by the alignment, "
                    "with the region ground truth held in a separate table."
                ),
            },
            "labelling": {
                "function": "sgv14_confirmatory_validation._labels",
                "exact_repair_outcome": OUTCOME_EXACT,
                "harm_policy": xc1_configuration["labelling"]["harm_policy"],
            },
            "evaluation_split": {"role": s14.ROLE_EVALUATION},
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        ENVIRONMENT_INVENTORY,
        {
            **_envelope("environment_inventory"),
            "environments": inventory,
            "totals": {
                key: sum(int(row[key]) for row in inventory)
                for key in (
                    "evaluation_documents",
                    "evaluation_images_present",
                    "evaluation_spans",
                    "evaluation_spans_with_box",
                    "ocr_error_sites",
                    "expressible_error_sites",
                )
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"freeze: {len(inventory)} environments -> {_relative(ENVIRONMENT_INVENTORY)}")
    return 0


# ------------------------------------------------------------------ sections 7-16, 50-57: design


def run_preregister() -> int:
    """Freeze the design before a single label exists."""
    started = time.monotonic()
    _require(FROZEN_CONFIGURATION, "freeze")
    _forbid(DESIGN_RECORD)
    if _labels_exist():
        raise PhaseError("labels already exist; the design cannot be registered after the fact")
    freeze = cc_read_json(RESEARCH_FREEZE)
    _write_json_once(
        DESIGN_RECORD,
        {
            **_envelope("design_record"),
            "head": _git("rev-parse", "HEAD"),
            "stage_kind": "DEVELOPMENT / MECHANISM",
            "question": (
                "if the location of a real OCR error is already known but the correct replacement "
                "is hidden, can stronger contemporary text or multimodal models generate the exact "
                "repair often enough to materially raise the correction opportunity ceiling?"
            ),
            "central_quantity": "P(exact correct replacement | real error location known)",
            "non_goals": [
                "a deployment, localization or reliability-ranking study",
                "threshold tuning, calibration, certification or human-review optimization",
                "retraining the downstream reliability model",
                "external confirmation or any use of the confirmatory reserve",
            ],
            "oracle_localization": {
                "analysis_only": True,
                "oracle_localization": True,
                "deployable": False,
                "location_source": "the SGV14 alignment, through SGV-XC1's _oracle_requests",
                "what_the_marker_reveals": (
                    "which OCR tokens the aligned error component covers (a span marker), or that "
                    "text is missing between two OCR tokens (a gap marker). In this population "
                    "every gap site is an omission and every omission is a gap, so the gap marker "
                    "reveals omission by geometry; a span marker does not distinguish "
                    "substitution, spurious insertion and segmentation. The marker carries no "
                    "ground-truth character, length, token count, type or outcome."
                ),
            },
            "baseline": {
                "frozen_xc1_arms": list(XC1_BASELINES),
                "condition": XC1_ORACLE,
                "source": _relative(xc1.ORACLE_CANDIDATES),
                "note": (
                    "M0, M1 and M2 are SGV-XC1's oracle-localized arms, read from its labelled "
                    "table and re-scored at K=1/3/5 on the GEN1 sample. They keep their own "
                    "contracts; the reproduction gate requires their SGV-XC1 quantities to "
                    "reproduce exactly."
                ),
                "sgv_xc1_oracle_localization_by_method": freeze["sgv_xc1"][
                    "oracle_localization_by_method"
                ],
            },
            "models": {
                model: {k: v for k, v in spec.items() if k != "batch_size"}
                for model, spec in MODEL_SPECS.items()
            },
            "model_sequencing": (
                "this machine has 16 GiB of unified memory, too little free disk for all three new "
                "checkpoints at once, and a download rate that made fetching every checkpoint "
                "before any generation impractical. Models are therefore downloaded, probed, "
                "audited and run one or two at a time: each model's probe and GT-blind format "
                "audit are written before that model generates anything; the prompt, decoding, "
                "contexts and sample are frozen before the first model generates; and the "
                "availability audit and model versions are assembled from the per-model records "
                "before any label exists. A checkpoint may be removed once its outputs and its "
                "regeneration check are cached; every file is hashed when probed, and the pinned "
                "revision makes a re-download byte-identical."
            ),
            "evidence_conditions": {
                E0: (
                    f"the marked OCR region with {CONTEXT_CHARS} OCR characters on each side, "
                    "line breaks in the context flattened to spaces (SGV-XC1's oracle context)"
                ),
                E1: "the OCR line or lines holding the site's anchor spans, marker in place",
                E2: "E1 plus the OCR line immediately before and after it on the same page",
                E3: "E1 text plus a deterministic crop of the page image around the site",
            },
            "cells": [list(cell) for cell in PRIMARY_CELLS],
            "control_cells": [
                {"model": m, "condition": c, "reference": r} for m, c, r in CONTROL_CELLS
            ],
            "candidate_budget": {
                "max_candidates": MAX_CANDIDATES,
                "k_grid": list(K_GRID),
                "primary_k": PRIMARY_K,
                "note": (
                    "sections 14 and 55: every GEN1 arm returns at most five ranked replacements "
                    "in one greedy decode, so no arm has more guesses than another. K=3 and K=5 "
                    "are prefixes of that list and are reported as ranking headroom only."
                ),
            },
            "metrics": {
                "exact_generation_at_k": (
                    "evaluable known error sites whose top-K distinct candidates include an exact "
                    "repair / evaluable known error sites in the sample"
                ),
                "evaluable": (
                    "an expressible OCR error site whose region SGV14's labeller can label -- "
                    "the set on which SGV-XC1's ground-truth corrector reached 1.0"
                ),
                "exact_repair_outcome": OUTCOME_EXACT,
                "abstention": "a valid answer with no candidate, or only the unchanged OCR text",
                "candidate_precision": "exact candidates / all proposed candidates",
            },
            "sample": {
                "unit": "known error site, drawn per environment",
                "order": "sha256 of the stage seed and the site id -- content-blind",
                "seed": GEN1_SEED,
                "per_environment": SAMPLE_PER_ENVIRONMENT,
                "rule": (
                    "the first sites of each environment in that order, or every site where fewer "
                    "exist. A constant, fixed from the GT-blind M2G audit's measured throughput "
                    "before any evaluation output existed."
                ),
                "control_subsample_per_environment": CONTROL_PER_ENVIRONMENT,
                "control_subsample": "the first sites of the same order, so it nests in the sample",
            },
            "reissues": [
                {
                    "superseded": _relative(path),
                    "superseded_sha256": file_sha256(path),
                    "reason": reason,
                }
                for path, reason in zip(
                    (SUPERSEDED_DESIGN, SUPERSEDED_DESIGN_V1), REISSUE_REASONS, strict=True
                )
                if path.is_file()
            ],
            "thresholds": {
                "meaningful_delta": MEANINGFUL_DELTA,
                "weak_type_delta": WEAK_TYPE_DELTA,
                "weak_error_types": list(WEAK_ERROR_TYPES),
                "weak_types_required": WEAK_TYPES_REQUIRED,
                "breadth_majority": BREADTH_MAJORITY,
                "environment_count": ENVIRONMENT_COUNT,
                "alpha": ALPHA,
                "note": (
                    "compatible with SGV-XC1's margin logic: an absolute 0.10 gain, measured at "
                    "K=1 against a named reference cell. Any positive delta is not success."
                ),
            },
            "effects": {
                "capacity": {name: [list(a), list(b)] for name, a, b in CAPACITY_EFFECTS},
                "evidence": {name: [list(a), list(b)] for name, a, b in EVIDENCE_EFFECTS},
                "joint": {name: [list(a), list(b)] for name, a, b in JOINT_EFFECTS},
            },
            "questions": QUESTIONS,
            "outcome_taxonomy": OUTCOME_TAXONOMY,
            "outcome_rule": OUTCOME_RULE,
            "ready_for_reliability_transfer_rule": (
                "true only for outcome A, B or C: a frozen configuration whose K=1 gain is "
                "material, which already requires breadth and weak-type recovery"
            ),
            "controls": {
                "c_gt_corrector": "SGV-XC1's C3 on the sample: analysis only, uses GT replacement",
                "c_identity": "the unchanged OCR region at every site",
                "c_random_replacement": (
                    f"{MAX_CANDIDATES} seeded single-character edits of the OCR region per site "
                    "(SGV-XC1's perturbation), or seeded short strings at a gap"
                ),
                X_MARK: (
                    "the marker moved to the nearest clean OCR position; scored at the true site"
                ),
                X_CTX: "every context line taken from another document's site; the site kept",
                X_IMG: "the E3 crop replaced by another document's crop; the text kept",
            },
            "statistics": {
                "inference_unit": "document cluster, within environment",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "interval": "percentile",
                "primary_family": [name for name, _a, _b in PRIMARY_COMPARISONS] + [P4_NAME],
                "p4_rule": (
                    "the strongest GEN1 cell against M0's XC1 known-site top-1, with the "
                    "strongest re-selected in every resample, so the interval prices the selection"
                ),
                "multiplicity": "Holm over the primary family; each secondary family within itself",
            },
            "external_confirmation": {
                "ready_for_external_confirmation": False,
                "reason": (
                    "every arm uses oracle localization on exposed development environments. "
                    "Fixed in advance; it does not depend on the result."
                ),
                "confirmatory_reserve_consumed": False,
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"preregister: {len(PRIMARY_CELLS)} cells, {len(CONTROL_CELLS)} control cells, "
        f"K={list(K_GRID)} (primary {PRIMARY_K}), margin {MEANINGFUL_DELTA} -> "
        f"{_relative(DESIGN_RECORD)}"
    )
    return 0


# ------------------------------------------------------------------ sections 10, 12: the contract


def mark_text(text: str, start: int, end: int, gap: bool) -> str:
    """Insert the neutral marker. It reveals where, never what."""
    if not gap:
        return f"{text[:start]}{ERR_OPEN}{text[start:end]}{ERR_CLOSE}{text[end:]}"
    separator = text[start:end]
    if not separator:
        middle = ERR_GAP
    elif "\n" in separator:
        middle = f" {ERR_GAP}\n"
    else:
        middle = f" {ERR_GAP} "
    return f"{text[:start]}{middle}{text[end:]}"


def e0_text(before: str, region: str, after: str, gap: bool) -> str:
    """E0: SGV-XC1's oracle context. Context line breaks flattened; the region kept verbatim."""
    left = before.replace("\n", " ")
    right = after.replace("\n", " ")
    shown = " " if gap and region else region
    return mark_text(f"{left}{shown}{right}", len(left), len(left) + len(shown), gap)


def anchors_of(anchor_ref: str) -> list[str]:
    return [part for part in str(anchor_ref).split("\0")[1:] if part]


def line_window(page: Page, anchors: Sequence[str], neighbours: bool) -> tuple[int, int]:
    """The stream range of the site's lines, optionally widened by one line on each side."""
    indices = [page.span_line[a] for a in anchors]
    low, high = min(indices), max(indices)
    if neighbours:
        low, high = max(0, low - 1), min(len(page.lines) - 1, high + 1)
    return page.lines[low][0], page.lines[high][1]


def line_text(
    page: Page, anchors: Sequence[str], start: int, end: int, gap: bool, wide: bool
) -> str:
    """E1, or E2 when ``wide``: an exact slice of the OCR stream with the marker in place."""
    window_start, window_end = line_window(page, anchors, wide)
    if not (window_start <= start <= end <= window_end):
        raise PhaseError("a site lies outside its own lines; the line index is inconsistent")
    text = page.stream[window_start:window_end]
    return mark_text(text, start - window_start, end - window_start, gap)


def user_message(condition: str, text: str) -> str:
    return USER_TEMPLATES[TEMPLATE_OF[condition]].format(text=text)


def crop_box(page: Page, anchors: Sequence[str]) -> tuple[int, int, int, int] | None:
    """Section 37: the anchor spans, one OCR neighbour on each side within the line, padded.

    Built from OCR boxes alone. A gap's anchors are its two flanking spans, so a gap crop is the
    deterministic flanking box the brief asks for. No ground-truth character sizes it.
    """
    ordered = sorted(anchors, key=lambda a: page.order.index(a))
    first, last = page.order.index(ordered[0]), page.order.index(ordered[-1])
    chosen = list(ordered)
    if first > 0 and page.span_line[page.order[first - 1]] == page.span_line[ordered[0]]:
        chosen.append(page.order[first - 1])
    if (
        last + 1 < len(page.order)
        and page.span_line[page.order[last + 1]] == page.span_line[ordered[-1]]
    ):
        chosen.append(page.order[last + 1])
    boxes = [page.span_box[s] for s in chosen if page.span_box.get(s) is not None]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes if b is not None)
    y0 = min(b[1] for b in boxes if b is not None)
    x1 = max(b[2] for b in boxes if b is not None)
    y1 = max(b[3] for b in boxes if b is not None)
    pad = max(CROP_MIN_PAD, CROP_PAD_FRACTION * (y1 - y0))
    left = max(0, int(np.floor(x0 - pad)))
    top = max(0, int(np.floor(y0 - pad)))
    right = min(page.width, int(np.ceil(x1 + pad)))
    bottom = min(page.height, int(np.ceil(y1 + pad)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def render_crop(image: Any, box: tuple[int, int, int, int]) -> Any:
    """The frozen resize: height CROP_HEIGHT, aspect kept, width capped at CROP_MAX_WIDTH."""
    from PIL import Image

    region = image.crop(box)
    width, height = region.size
    target_w, target_h = max(1, round(width * CROP_HEIGHT / height)), CROP_HEIGHT
    if target_w > CROP_MAX_WIDTH:
        target_w, target_h = CROP_MAX_WIDTH, max(1, round(height * CROP_MAX_WIDTH / width))
    return region.resize((target_w, target_h), Image.Resampling.LANCZOS)


def parse_candidates(text: str, finished: bool) -> dict[str, Any]:
    """Section 12's contract, read deterministically. It reads no label.

    ``strict`` is the whole answer as one JSON object; ``lenient`` is a JSON object found after
    stripping a code fence or leading prose; ``truncated`` is an unfinished object from a decode
    that hit its cap, which keeps the candidate strings it completed and nothing else; anything
    else is ``invalid``. Candidates are stripped, de-duplicated in first-occurrence order and cut
    at MAX_CANDIDATES, and every one of those steps is counted.
    """
    record: dict[str, Any] = {
        "status": "invalid",
        "candidates": [],
        "object_items": 0,
        "non_string_items": 0,
        "duplicates_removed": 0,
        "over_limit": 0,
        "salvaged_from_truncation": 0,
    }
    stripped = text.strip()
    payload: Any = None
    try:
        payload = json.loads(stripped)
        record["status"] = "strict" if isinstance(payload, dict) else "invalid"
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict):
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped)
        opening = body.find("{")
        payload = None
        if opening >= 0:
            try:
                payload, _end = json.JSONDecoder().raw_decode(body[opening:])
            except json.JSONDecodeError:
                payload = None
        if isinstance(payload, dict):
            record["status"] = "lenient"
        elif opening >= 0 and not finished:
            record["status"] = "truncated"
            payload = {"candidates": _salvage(body[opening:])}
            record["salvaged_from_truncation"] = len(payload["candidates"])
        else:
            record["status"] = "invalid"
            return record
    items = payload.get("candidates")
    if not isinstance(items, list):
        record["status"] = "invalid"
        return record
    seen: list[str] = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("replacement"), str):
            record["object_items"] += 1
            item = item["replacement"]
        if not isinstance(item, str):
            record["non_string_items"] += 1
            continue
        value = item.strip()
        if value in seen:
            record["duplicates_removed"] += 1
            continue
        seen.append(value)
    record["over_limit"] = max(0, len(seen) - MAX_CANDIDATES)
    record["candidates"] = seen[:MAX_CANDIDATES]
    return record


def _salvage(body: str) -> list[str]:
    """The complete strings of a candidates list a decode cap cut off. Nothing is completed."""
    match = re.search(r'"candidates"\s*:\s*\[', body)
    if match is None:
        return []
    decoder = json.JSONDecoder()
    position = match.end()
    found: list[str] = []
    while True:
        while position < len(body) and body[position] in " \t\r\n,":
            position += 1
        if position >= len(body) or body[position] != '"':
            return found
        try:
            value, position = decoder.raw_decode(body, position)
        except json.JSONDecodeError:
            return found
        found.append(str(value))


def output_flags(text: str) -> dict[str, bool]:
    """GT-blind properties of one raw answer (section 35)."""
    control = (
        any(unicodedata.category(ch) == "Cc" and ch not in "\n\r\t" for ch in text) or "�" in text
    )
    return {
        "invalid_unicode": control,
        "echoed_prompt": ERR_OPEN in text
        or ERR_GAP in text
        or "OCR line:" in text
        or "OCR text" in text,
    }


def span_escape(candidate: str, region: str, before: str, after: str, gap: bool) -> bool:
    """Section 36: a candidate that carries unmarked neighbouring text, or a marker, with it.

    The neighbours are the whitespace-delimited OCR words either side of the site. One-character
    words are ignored, so a candidate is not flagged for sharing a letter with its context.
    """
    if any(tag in candidate for tag in (ERR_OPEN, ERR_CLOSE, ERR_GAP)):
        return True
    if "\n" in candidate and "\n" not in region:
        return True
    left_words, right_words = before.split(), after.split()
    left = left_words[-1] if left_words else ""
    right = right_words[0] if right_words else ""
    body = region.strip()
    if len(left) >= 2 and candidate.startswith(left + " ") and not body.startswith(left):
        return True
    if len(right) >= 2 and candidate.endswith(" " + right) and not body.endswith(right):
        return True
    return gap and (
        (len(left) >= 2 and candidate.startswith(left))
        or (len(right) >= 2 and candidate.endswith(right))
    )


# ------------------------------------------------------------------ sections 8, 15: the runners


def _generation_settings(tokenizer: Any) -> dict[str, Any]:
    return {
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "repetition_penalty": 1.0,
        "pad_token_id": tokenizer.pad_token_id,
    }


def _eos_ids(model: Any, tokenizer: Any) -> set[int]:
    ids = model.generation_config.eos_token_id
    found = set(ids if isinstance(ids, list) else [ids] if ids is not None else [])
    if tokenizer.eos_token_id is not None:
        found.add(int(tokenizer.eos_token_id))
    return {int(i) for i in found}


def _decode_rows(
    generated: Any, width: int, tokenizer: Any, eos: set[int]
) -> list[tuple[str, int, bool]]:
    rows: list[tuple[str, int, bool]] = []
    for sequence in generated[:, width:].tolist():
        stop = next((i for i, token in enumerate(sequence) if token in eos), None)
        kept = sequence if stop is None else sequence[:stop]
        kept = [t for t in kept if t != tokenizer.pad_token_id]
        rows.append(
            (str(tokenizer.decode(kept, skip_special_tokens=True)), len(kept), stop is not None)
        )
    return rows


class _TextRunner:
    """A pinned decoder-only instruct model. Frozen prompt, greedy decode, local weights only."""

    def __init__(self, snapshot: Path, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        # Decoder-only batching pads on the left, or every short row's continuation is wrong.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model: Any = AutoModelForCausalLM.from_pretrained(
            str(snapshot), local_files_only=True, dtype=getattr(torch, MODEL_DTYPE)
        )
        model.eval()
        self.model = model.to(device)
        self.device = device
        self.eos = _eos_ids(self.model, self.tokenizer)
        torch.set_grad_enabled(False)

    def render(self, system: str, user: str) -> str:
        return str(
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def generate(
        self, batch: Sequence[tuple[str, str, str | None]], max_new_tokens: int
    ) -> list[tuple[str, int, int, bool]]:
        import torch

        chats = [self.render(system, user) for system, user, _image in batch]
        encoded = self.tokenizer(chats, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded, max_new_tokens=max_new_tokens, **_generation_settings(self.tokenizer)
            )
        width = int(encoded["input_ids"].shape[1])
        prompt_tokens = encoded["attention_mask"].sum(dim=1).tolist()
        rows = _decode_rows(generated, width, self.tokenizer, self.eos)
        del encoded, generated
        _release_device_cache(self.device)
        return [
            (text, int(p), n, done) for (text, n, done), p in zip(rows, prompt_tokens, strict=True)
        ]


class _VisionRunner:
    """The pinned image-text model. With no image it is its own matched text-only comparator."""

    def __init__(self, snapshot: Path, device: str) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model: Any = AutoModelForImageTextToText.from_pretrained(
            str(snapshot), local_files_only=True, dtype=getattr(torch, MODEL_DTYPE)
        )
        model.eval()
        self.model = model.to(device)
        self.device = device
        self.eos = _eos_ids(self.model, self.tokenizer)
        torch.set_grad_enabled(False)

    def render(self, system: str, user: str, image: bool) -> str:
        content: list[dict[str, str]] = [{"type": "image"}] if image else []
        content.append({"type": "text", "text": user})
        return str(
            self.processor.apply_chat_template(
                [
                    {"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def generate(
        self, batch: Sequence[tuple[str, str, str | None]], max_new_tokens: int
    ) -> list[tuple[str, int, int, bool]]:
        import torch
        from PIL import Image

        with_images = [image is not None for _s, _u, image in batch]
        if any(with_images) and not all(with_images):
            raise PhaseError("a batch mixes image and text-only requests")
        texts = [self.render(system, user, image is not None) for system, user, image in batch]
        images = (
            [Image.open(image).convert("RGB") for _s, _u, image in batch if image is not None]
            if all(with_images)
            else None
        )
        encoded = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded, max_new_tokens=max_new_tokens, **_generation_settings(self.tokenizer)
            )
        width = int(encoded["input_ids"].shape[1])
        prompt_tokens = encoded["attention_mask"].sum(dim=1).tolist()
        rows = _decode_rows(generated, width, self.tokenizer, self.eos)
        del encoded, generated
        _release_device_cache(self.device)
        return [
            (text, int(p), n, done) for (text, n, done), p in zip(rows, prompt_tokens, strict=True)
        ]


def _runner(model: str, device: str) -> Any:
    snapshot = _snapshot(model)
    if snapshot is None:
        raise PhaseError(f"{model}: no local snapshot at {_model_dir(model)}")
    return (
        _VisionRunner(snapshot, device)
        if MODEL_SPECS[model]["kind"] == "vision"
        else _TextRunner(snapshot, device)
    )


def _allowance(region: str, gap: bool) -> int:
    return GAP_ALLOWANCE_CHARS if gap else len(region)


def max_new_tokens(widest_chars: int) -> int:
    """Section 15's output cap for one batch, from its widest region alone."""
    per_candidate = -(-widest_chars // CHARS_PER_TOKEN) + GEN_PER_CANDIDATE_MARGIN
    return min(GEN_MAX_NEW_TOKENS, GEN_BASE_TOKENS + MAX_CANDIDATES * per_candidate)


def run_requests(
    runner: Any,
    requests: pd.DataFrame,
    batch_size: int,
    progress: Callable[[str], None],
    *,
    limit_batches: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One cell over one set of requests. Raw answers exactly as decoded, plus cost accounting.

    Requests run sorted by prompt length then request id, so a batch pads little; the output cap
    is set from the widest region in the batch. Both are part of the arm's definition: a bf16
    decode on this device is reproducible for fixed batching and is not invariant to it.
    """
    frame = requests.reset_index(drop=True)
    prompts = [str(p) for p in frame["user_prompt"]]
    images = [None if pd.isna(p) or not p else str(p) for p in frame["image_path"]]
    ids = frame["request_id"].astype(str).tolist()
    allowances = np.array(
        [
            _allowance(str(r), str(k) == "gap")
            for r, k in zip(frame["region"], frame["anchor_kind"], strict=True)
        ]
    )
    order = sorted(range(len(frame)), key=lambda i: (len(prompts[i]), ids[i]))
    began = time.monotonic()
    rows: list[dict[str, Any]] = []
    peak = 0
    batches = 0
    for start in range(0, len(order), batch_size):
        if limit_batches is not None and batches >= limit_batches:
            break
        chunk = order[start : start + batch_size]
        cap = max_new_tokens(int(allowances[chunk].max()))
        outputs = runner.generate([(SYSTEM_PROMPT, prompts[i], images[i]) for i in chunk], cap)
        for i, (text, prompt_tokens, output_tokens, finished) in zip(chunk, outputs, strict=True):
            rows.append(
                {
                    "environment": str(frame.at[i, "environment"]),
                    "method": str(frame.at[i, "method"]),
                    "condition": str(frame.at[i, "condition"]),
                    "request_id": ids[i],
                    "rank": 0,
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
        try:
            import torch

            peak = max(peak, int(torch.mps.driver_allocated_memory()))
        except Exception:
            peak = 0
        if batches % 10 == 0:
            rate = len(rows) / max(time.monotonic() - began, 1e-9)
            progress(f"    {len(rows)}/{len(frame)} ({rate:.2f}/s)")
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
        "decoding_revision": DECODING_REVISION,
        "max_new_tokens_rule": MAX_NEW_TOKENS_RULE,
        "peak_device_memory_bytes": peak or None,
    }
    return raw, cost


# ------------------------------------------------------------------ section 9: probe and audit


def _selected_models() -> list[str]:
    chosen = [m for m in os.environ.get("GEN1_MODEL", "").split(",") if m]
    unknown = [m for m in chosen if m not in GEN1_MODELS]
    if unknown:
        raise PhaseError(f"unknown GEN1_MODEL entries: {unknown}")
    return [m for m in GEN1_MODELS if not chosen or m in chosen]


def _front_matter(readme: Path) -> dict[str, Any]:
    """The model card's YAML header, read verbatim. A card is evidence, not recollection."""
    if not readme.is_file():
        return {}
    text = readme.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    import yaml

    try:
        loaded = yaml.safe_load(text[3:end])
    except yaml.YAMLError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _card_lines(readme: Path, word: str, limit: int = 4) -> list[str]:
    if not readme.is_file():
        return []
    lines = readme.read_text(encoding="utf-8", errors="replace").splitlines()
    found = [line.strip()[:240] for line in lines if word in line.lower() and line.strip()]
    return found[:limit]


def _vision_parameters(snapshot: Path) -> float | None:
    """Parameters in the vision tower, read from the safetensors headers like the total."""
    import struct

    total = 0
    tensors = sorted(snapshot.glob("*.safetensors"))
    for path in tensors:
        with path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        for name, spec in header.items():
            if name == "__metadata__" or not isinstance(spec, dict):
                continue
            if name.startswith(("visual.", "model.visual.")):
                count = 1
                for dimension in spec.get("shape") or []:
                    count *= int(dimension)
                total += count
    return round(total / 1e6, 3) if tensors else None


def run_probe() -> int:
    """Section 9, per model, while its checkpoint is on disk: what it is, hashed file by file."""
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    for model in _selected_models():
        path = PROBE_DIR / f"{model}.json"
        if path.exists():
            print(f"  {model}: probe exists")
            continue
        spec = MODEL_SPECS[model]
        snapshot = _snapshot(model)
        if snapshot is None:
            print(f"  {model}: no local snapshot at {_model_dir(model)}; not probed")
            continue
        began = time.monotonic()
        config = cc_read_json(snapshot / "config.json")
        text_config = config.get("text_config") or config
        parameters, source = _parameter_count(snapshot)
        card = _front_matter(snapshot / "README.md")
        weights = sorted(snapshot.glob("*.safetensors"))
        _write_json_once(
            path,
            {
                **_envelope("model_probe"),
                "model": model,
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                "local_or_api": "local",
                "snapshot": str(snapshot),
                "architecture": (config.get("architectures") or ["unknown"])[0],
                "model_type": str(config.get("model_type", "unknown")),
                "multimodal": "vision_config" in config,
                "parameters_millions": parameters,
                "vision_parameters_millions": (
                    _vision_parameters(snapshot) if "vision_config" in config else 0.0
                ),
                "parameter_count_source": source,
                "context_window_tokens": int(text_config.get("max_position_embeddings") or 0),
                "checkpoint_dtype": str(
                    text_config.get("torch_dtype") or config.get("torch_dtype") or ""
                ),
                "weight_bytes": sum(p.stat().st_size for p in weights),
                "expected_memory_note": (
                    "bf16 weights occupy weight_bytes of unified memory before activations"
                ),
                "card_license": card.get("license"),
                "card_languages": card.get("language"),
                "card_language_lines": _card_lines(snapshot / "README.md", "language"),
                "card_pipeline_tag": card.get("pipeline_tag"),
                "checkpoint_files": _checkpoint_files(snapshot),
                "custom_code_used": False,
                "probed_seconds": time.monotonic() - began,
            },
        )
        print(f"  {model}: {parameters}M parameters, {len(weights)} weight files hashed")
    return 0


def _audit_requests(spec: dict[str, Any]) -> pd.DataFrame:
    """ADAPTATION-split regions and gaps (SGV-XC1's audit sites); never an evaluation page."""
    name = spec["environment"]
    m0 = pd.read_parquet(xc1.CACHE / f"{_slug(name)}.m0_candidates.parquet")
    sites = (
        m0[m0["role"] == s14.ROLE_ADAPTATION]
        .drop_duplicates("site_id")
        .sort_values("site_id", kind="stable")
    )
    gap = sites["anchor_kind"].astype(str) == "gap"
    chosen = pd.concat(
        [sites[~gap].head(AUDIT_REGIONS), sites[gap].head(AUDIT_GAPS)], ignore_index=True
    )
    return pd.DataFrame(
        {
            "environment": name,
            "site_id": (name + "|audit-" + chosen["site_id"].astype(str)).to_numpy(),
            "document_id": chosen["document_id"].astype(str).to_numpy(),
            "engine_id": chosen["engine_id"].astype(str).to_numpy(),
            "anchor_kind": chosen["anchor_kind"].astype(str).to_numpy(),
            "anchor_ref": chosen["anchor_ref"].astype(str).to_numpy(),
            "char_start": chosen["char_start"].to_numpy(dtype=np.int64),
            "char_end": chosen["char_end"].to_numpy(dtype=np.int64),
            "region": chosen["original_ocr"].astype(str).to_numpy(),
            "context_before": chosen["context_before"].astype(str).to_numpy(),
            "context_after": chosen["context_after"].astype(str).to_numpy(),
        }
    )


def build_contexts(
    sites: pd.DataFrame, pages: dict[tuple[str, str], Page], crop_dir: Path
) -> pd.DataFrame:
    """E0, E1, E2 and the E3 crop for every site. GT-blind: OCR text, OCR boxes and the image.

    ``sites`` carries the GT-blind request columns only; the function never receives a truth
    column, and the leakage suite checks that its source names none.
    """
    from PIL import Image

    crop_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    images: dict[str, Any] = {}
    for row in sites.itertuples(index=False):
        page = pages[str(row.document_id), str(row.engine_id)]
        anchors = anchors_of(str(row.anchor_ref))
        gap = str(row.anchor_kind) == "gap"
        start, end = int(row.char_start), int(row.char_end)
        if page.stream[start:end] != str(row.region):
            raise PhaseError(f"{row.site_id}: region differs from the rebuilt OCR stream")
        box = crop_box(page, anchors)
        crop_path = crop_hash = None
        size: tuple[int, int] | None = None
        if box is not None and Path(page.image_path).is_file():
            if page.image_path not in images:
                images = {page.image_path: Image.open(page.image_path).convert("RGB")}
            crop = render_crop(images[page.image_path], box)
            target = crop_dir / f"{canonical_hash({'site': str(row.site_id)})[:32]}.png"
            if not target.is_file():
                crop.save(target, format="PNG")
            crop_path = str(target)
            crop_hash = canonical_hash(
                {"size": list(crop.size), "rgb": canonical_hash(crop.tobytes().hex())}
            )
            size = crop.size
        rows.append(
            {
                "site_id": str(row.site_id),
                "e0_text": e0_text(
                    str(row.context_before), str(row.region), str(row.context_after), gap
                ),
                "e1_text": line_text(page, anchors, start, end, gap, wide=False),
                "e2_text": line_text(page, anchors, start, end, gap, wide=True),
                "crop_path": crop_path,
                "crop_sha256": crop_hash,
                "crop_box": list(box) if box is not None else None,
                "crop_width": size[0] if size else None,
                "crop_height": size[1] if size else None,
                "page_image_sha256": page.image_sha256,
            }
        )
    return pd.DataFrame(rows)


def _cell_frame(
    base: pd.DataFrame, contexts: pd.DataFrame, model: str, condition: str, prefix: str
) -> pd.DataFrame:
    """Requests for one (model, condition): the prompt text, the image if any, the site columns."""
    joined = base.merge(contexts, on="site_id", how="inner", validate="one_to_one")
    template = TEMPLATE_OF[condition]
    column = {E0: "e0_text", E1: "e1_text", E2: "e2_text", E3: "e1_text"}[template]
    if condition in (X_MARK, X_CTX):
        column = "control_text"
    image = template == E3
    if image:
        joined = joined[joined["crop_path"].notna()]
    digest = joined["site_id"].astype(str).str.split("|", n=1).str[1]
    return pd.DataFrame(
        {
            "environment": joined["environment"].to_numpy(),
            "method": model,
            "condition": condition,
            "request_id": (joined["environment"] + f"|{prefix}{condition}|" + digest).to_numpy(),
            "site_id": joined["site_id"].to_numpy(),
            "anchor_kind": joined["anchor_kind"].to_numpy(),
            "region": joined["region"].to_numpy(),
            "user_prompt": [user_message(condition, str(t)) for t in joined[column]],
            "image_path": joined["crop_path"].to_numpy() if image else None,
            "image_sha256": joined["crop_sha256"].to_numpy() if image else None,
        }
    )


def _format_census(requests: pd.DataFrame, raw: pd.DataFrame) -> dict[str, Any]:
    """GT-blind measurements of one set of answers: the contract, abstention and span escape."""
    by_id = requests.set_index("request_id")
    counts = {
        "answers": 0,
        "strict": 0,
        "lenient": 0,
        "truncated": 0,
        "invalid": 0,
        "empty_list": 0,
        "identity_only": 0,
        "with_a_proposal": 0,
        "candidates": 0,
        "span_escape_candidates": 0,
        "echoed_prompt": 0,
        "invalid_unicode": 0,
        "over_limit": 0,
    }
    for row in raw.itertuples(index=False):
        request = by_id.loc[str(row.request_id)]
        parsed = parse_candidates(str(row.output), bool(row.finished))
        flags = output_flags(str(row.output))
        counts["answers"] += 1
        counts[str(parsed["status"])] += 1
        counts["echoed_prompt"] += int(flags["echoed_prompt"])
        counts["invalid_unicode"] += int(flags["invalid_unicode"])
        counts["over_limit"] += int(parsed["over_limit"] > 0)
        region = str(request["region"])
        gap = str(request["anchor_kind"]) == "gap"
        proposals = [c for c in parsed["candidates"] if c.strip() != region.strip()]
        if parsed["status"] in ("strict", "lenient"):
            if not parsed["candidates"]:
                counts["empty_list"] += 1
            elif not proposals:
                counts["identity_only"] += 1
            else:
                counts["with_a_proposal"] += 1
        counts["candidates"] += len(proposals)
        counts["span_escape_candidates"] += sum(
            span_escape(
                c, region, str(request["context_before"]), str(request["context_after"]), gap
            )
            for c in proposals
        )
    total = max(counts["answers"], 1)
    return {
        **counts,
        "valid_share": (counts["strict"] + counts["lenient"]) / total,
        "strict_share": counts["strict"] / total,
        "abstention_share": (counts["empty_list"] + counts["identity_only"]) / total,
        "mean_candidates_per_answer": counts["candidates"] / total,
    }


def run_audit() -> int:
    """A GT-blind format, throughput and determinism audit, on ADAPTATION documents only.

    It checks -- before the design is sized and the registries freeze, and before any label exists
    -- whether the frozen contract elicits parseable answers, how fast each cell runs, and whether a
    repeat reproduces it. It never touches an evaluation page and never reads a label, so it cannot
    tune an arm toward an outcome. The prompt is revised at most once, whatever this shows.
    """
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    if _labels_exist():
        raise PhaseError("labels already exist; the audit belongs before them")
    import transformers

    transformers.utils.logging.set_verbosity_error()
    device = _resolve_device()
    silent: Callable[[str], None] = lambda _message: None  # noqa: E731
    base_frames: list[pd.DataFrame] = []
    context_frames: list[pd.DataFrame] = []
    for name in AUDIT_ENVIRONMENTS:
        spec = _spec(name)
        requests = _audit_requests(spec)
        pages = _page_index(spec)
        base_frames.append(requests)
        context_frames.append(build_contexts(requests, pages, CACHE / "audit_crops" / _slug(name)))
    base = pd.concat(base_frames, ignore_index=True)
    contexts = pd.concat(context_frames, ignore_index=True)
    for model in _selected_models():
        path = AUDIT_DIR / f"{model}.json"
        if path.exists():
            print(f"  {model}: audit exists")
            continue
        if _snapshot(model) is None:
            print(f"  {model}: no local snapshot; not audited")
            continue
        began = time.monotonic()
        runner = _runner(model, device)
        batch = int(MODEL_SPECS[model]["batch_size"])
        conditions: dict[str, Any] = {}
        first_requests = first_raw = None
        for condition in CELL_CONDITIONS[model]:
            requests = _cell_frame(base, contexts, model, condition, "audit-")
            requests = requests.merge(
                base[["site_id", "context_before", "context_after"]], on="site_id"
            )
            raw, cost = run_requests(runner, requests, batch, silent)
            conditions[condition] = {
                **_format_census(requests, raw),
                "requests_per_second": cost["requests_per_second"],
                "prompt_tokens": cost["prompt_tokens"],
                "output_tokens": cost["output_tokens"],
                "unfinished_decodes": cost["unfinished_decodes"],
                "peak_device_memory_bytes": cost["peak_device_memory_bytes"],
            }
            if first_requests is None:
                first_requests, first_raw = requests, raw
            print(
                f"  {model}/{condition}: valid {conditions[condition]['valid_share']:.2f} "
                f"abstain {conditions[condition]['abstention_share']:.2f} "
                f"{cost['requests_per_second']:.2f} req/s",
                flush=True,
            )
        assert first_requests is not None and first_raw is not None
        repeat, _cost = run_requests(runner, first_requests, batch, silent)
        halved, _cost = run_requests(runner, first_requests, max(1, batch // 2), silent)
        first = dict(zip(first_raw["request_id"], first_raw["output"], strict=True))
        again = dict(zip(repeat["request_id"], repeat["output"], strict=True))
        half = dict(zip(halved["request_id"], halved["output"], strict=True))
        del runner
        _release_device_cache(device)
        _write_json_once(
            path,
            {
                **_envelope("prompt_format_audit_model"),
                "model": model,
                "split": s14.ROLE_ADAPTATION,
                "environments": list(AUDIT_ENVIRONMENTS),
                "ground_truth_read": False,
                "evaluation_documents_used": False,
                "prompt_revision": PROMPT_REVISION,
                "prompt_hash": canonical_hash({"system": SYSTEM_PROMPT, "user": USER_TEMPLATES}),
                "decoding_revision": DECODING_REVISION,
                "max_new_tokens_rule": MAX_NEW_TOKENS_RULE,
                "allocator_environment": _allocator_environment(),
                "batch_size": batch,
                "requests_per_condition": len(first_requests),
                "gap_requests_per_condition": int((first_requests["anchor_kind"] == "gap").sum()),
                "conditions": conditions,
                "repeat_condition": CELL_CONDITIONS[model][0],
                "identical_on_repeat": first == again,
                "answers_differing_at_half_batch": sum(first[k] != half.get(k) for k in first),
                "audit_seconds": time.monotonic() - began,
            },
        )
        print(f"  {model}: audit written, repeat identical {first == again}", flush=True)
    return 0


def run_availability() -> int:
    """Section 9: the availability audit, assembled once every model has been probed and audited.

    The primary model family is frozen here. A model without a probe and an audit is recorded as
    unavailable with the reason, and nothing may be substituted for it afterwards.
    """
    started = time.monotonic()
    for path in (MODEL_AVAILABILITY, PROMPT_FORMAT_AUDIT, MODEL_VERSIONS):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("labels already exist; availability is recorded before them")
    models: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for model in GEN1_MODELS:
        spec = MODEL_SPECS[model]
        probe_path, audit_path = PROBE_DIR / f"{model}.json", AUDIT_DIR / f"{model}.json"
        probe = cc_read_json(probe_path) if probe_path.is_file() else None
        audit = cc_read_json(audit_path) if audit_path.is_file() else None
        valid = (
            min(float(c["valid_share"]) for c in audit["conditions"].values()) if audit else None
        )
        available = probe is not None and audit is not None
        models[model] = {
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "kind": spec["kind"],
            "family": spec["family"],
            "developer": spec["developer"],
            "role": spec["role"],
            "reason_for_inclusion": spec["reason_for_inclusion"],
            "local_or_api": "local",
            "multimodal": bool(probe["multimodal"]) if probe else spec["kind"] == "vision",
            "multilingual_evidence": {
                "card_languages": probe["card_languages"] if probe else None,
                "card_language_lines": probe["card_language_lines"] if probe else [],
            },
            "license": probe["card_license"] if probe else None,
            "parameters_millions": probe["parameters_millions"] if probe else None,
            "vision_parameters_millions": probe["vision_parameters_millions"] if probe else None,
            "context_window_tokens": probe["context_window_tokens"] if probe else None,
            "expected_memory_bytes": probe["weight_bytes"] if probe else None,
            "expected_requests_per_second": (
                {c: float(v["requests_per_second"]) for c, v in audit["conditions"].items()}
                if audit
                else None
            ),
            "lowest_valid_answer_share_in_audit": valid,
            "reproducibility": (
                "local pinned revision, every file hashed, greedy decode; identical on repeat "
                "in the audit"
                if audit and audit["identical_on_repeat"]
                else "not established"
            ),
            "available": available,
            "availability_outcome": "available" if available else "unavailable",
            "reason": None
            if available
            else "no probe and audit could be completed on this machine",
            "probe": _relative(probe_path) if probe else None,
            "audit": _relative(audit_path) if audit else None,
        }
        if audit is not None:
            audits[model] = audit
    unavailable = [m for m, record in models.items() if not record["available"]]
    _write_json_once(
        MODEL_AVAILABILITY,
        {
            **_envelope("model_availability_audit"),
            "models": models,
            "primary_family": list(GEN1_MODELS),
            "available_family": [m for m in GEN1_MODELS if models[m]["available"]],
            "unavailable_family": unavailable,
            "declared_unavailable_candidates": {
                "gated_checkpoints": (
                    "Gemma 3 and Llama 3.2 Vision require an access token; none is configured, so "
                    "neither was considered available. Recorded before any endpoint."
                ),
                "larger_checkpoints": (
                    "7B-and-larger bf16 checkpoints exceed this machine's 16 GiB of unified "
                    "memory once the OS and the page images are resident."
                ),
                "external_apis": (
                    "not used: sending evaluation pages to an external service was not "
                    "authorized, and the brief prefers open weights."
                ),
            },
            "frozen_before_endpoint_evaluation": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        PROMPT_FORMAT_AUDIT,
        {
            **_envelope("prompt_format_audit"),
            "ground_truth_read": False,
            "evaluation_documents_used": False,
            "split": s14.ROLE_ADAPTATION,
            "prompt_revision": PROMPT_REVISION,
            "prompt_hash": canonical_hash({"system": SYSTEM_PROMPT, "user": USER_TEMPLATES}),
            "definitions": {
                "strict": "the whole answer is one JSON object with a candidates list",
                "lenient": "a JSON object found after a code fence or leading prose",
                "truncated": "an unfinished JSON object from a decode that hit its cap",
                "invalid": "anything else",
                "abstention": "a valid empty list, or candidates that only repeat the OCR text",
                "span_escape": "a candidate carrying an unmarked neighbouring word or a marker",
            },
            "models": audits,
            "decision_rule": (
                "the prompt is revised at most once, on the first audit, and frozen whatever the "
                "later audits show; an audit informs the record, never a search"
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    import platform

    import torch
    import transformers

    probes = {
        model: cc_read_json(PROBE_DIR / f"{model}.json")
        for model in GEN1_MODELS
        if (PROBE_DIR / f"{model}.json").is_file()
    }
    _write_json_once(
        MODEL_VERSIONS,
        {
            **_envelope("model_versions"),
            "models": {
                model: {
                    key: probe[key]
                    for key in (
                        "repo_id",
                        "revision",
                        "snapshot",
                        "architecture",
                        "model_type",
                        "multimodal",
                        "parameters_millions",
                        "vision_parameters_millions",
                        "parameter_count_source",
                        "context_window_tokens",
                        "weight_bytes",
                        "card_license",
                        "checkpoint_files",
                        "issued_utc",
                    )
                }
                for model, probe in probes.items()
            },
            "inference_dtype": MODEL_DTYPE,
            "packages": {
                "torch": str(torch.__version__),
                "transformers": str(transformers.__version__),
            },
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python": platform.python_version(),
                "device": _resolve_device(),
            },
            "external_api_calls": 0,
            "note": (
                "each model's entry is its probe, written while its checkpoint was on disk and "
                "before it generated; issued_utc is that probe's time"
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"availability: {len(GEN1_MODELS) - len(unavailable)} of {len(GEN1_MODELS)} models "
        f"available" + (f" (unavailable: {unavailable})" if unavailable else "")
    )
    return 0


# ------------------------------------------------------------------ sections 33, 34: the sites


def _xc1_population(spec: dict[str, Any], labelable: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SGV-XC1's oracle requests and truth for one environment, restricted to labelable sites."""
    slug = _slug(spec["environment"])
    requests = pd.read_parquet(xc1.CACHE / f"{slug}.oracle_requests.parquet")
    truth = pd.read_parquet(xc1.CACHE / f"{slug}.oracle_truth.parquet")
    keep = requests["site_id"].astype(str).isin(labelable)
    return requests[keep].reset_index(drop=True), truth[truth["site_id"].isin(labelable)]


def run_sites() -> int:
    """Sections 33 and 34: the evaluable population, its accounting, and the frozen sample."""
    started = time.monotonic()
    _require(DESIGN_RECORD, "preregister")
    for path in (ERROR_SITE_INVENTORY, SAMPLE_SITES, SAMPLE_TRUTH):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("labels already exist; the sample cannot be drawn after them")
    per_environment = SAMPLE_PER_ENVIRONMENT

    oracle = cc_read_json(xc1.ORACLE_LOCALIZATION)
    census = {row["environment"]: row for row in oracle["expressibility"]}
    c3 = pd.read_parquet(xc1.ORACLE_CANDIDATES, columns=["method", "site_id"])
    labelable = set(c3.loc[c3["method"] == C3_ORACLE, "site_id"].astype(str))
    sample_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    for spec in environment_specs():
        name = spec["environment"]
        slug = _slug(name)
        all_truth = pd.read_parquet(xc1.CACHE / f"{slug}.oracle_truth.parquet")
        requests, truth = _xc1_population(spec, labelable)
        order = requests["site_id"].astype(str).map(_order_key)
        ranked = requests.assign(_order=order.to_numpy()).sort_values("_order", kind="stable")
        chosen = ranked.head(per_environment).drop(columns=["_order"]).reset_index(drop=True)
        chosen["sample_rank"] = np.arange(len(chosen), dtype=np.int64)
        chosen["in_control_subsample"] = chosen["sample_rank"] < CONTROL_PER_ENVIRONMENT
        chosen_truth = truth[truth["site_id"].isin(set(chosen["site_id"]))].assign(environment=name)
        sample_frames.append(chosen)
        truth_frames.append(chosen_truth)
        kinds_all = all_truth["site_kind"].value_counts().to_dict()
        kinds_pop = truth["site_kind"].value_counts().to_dict()
        kinds_sample = chosen_truth["site_kind"].value_counts().to_dict()
        inventory.append(
            {
                "environment": name,
                "corpus": spec["corpus"],
                "base_engine": spec["base_engine"],
                "ocr_error_sites": int(census[name]["ocr_error_sites"]),
                "expressible": int(census[name]["expressible"]),
                "not_expressible": int(census[name]["ocr_error_sites"])
                - int(census[name]["expressible"]),
                "evaluable_labelable": len(requests),
                "expressible_not_labelable": int(census[name]["expressible"]) - len(requests),
                "sampled": len(chosen),
                "control_subsample": int(chosen["in_control_subsample"].sum()),
                "sampled_documents": int(chosen["document_id"].nunique()),
                "expressible_by_error_type": {str(k): int(v) for k, v in kinds_all.items()},
                "evaluable_by_error_type": {str(k): int(v) for k, v in kinds_pop.items()},
                "sampled_by_error_type": {str(k): int(v) for k, v in kinds_sample.items()},
                "sampled_by_anchor_kind": {
                    str(k): int(v) for k, v in chosen["anchor_kind"].value_counts().items()
                },
            }
        )
    sample = pd.concat(sample_frames, ignore_index=True)
    leaked = [c for c in FORBIDDEN_INPUT_COLUMNS if c in sample.columns]
    if leaked:
        raise PhaseError(f"the GT-blind sample table carries truth columns: {leaked}")
    _write_parquet_once(SAMPLE_SITES, sample)
    _write_parquet_once(SAMPLE_TRUTH, pd.concat(truth_frames, ignore_index=True))
    totals = {
        key: sum(int(row[key]) for row in inventory)
        for key in (
            "ocr_error_sites",
            "expressible",
            "not_expressible",
            "evaluable_labelable",
            "expressible_not_labelable",
            "sampled",
            "control_subsample",
            "sampled_documents",
        )
    }
    by_type: dict[str, int] = {}
    for row in inventory:
        for kind, count in row["sampled_by_error_type"].items():
            by_type[kind] = by_type.get(kind, 0) + count
    _write_json_once(
        ERROR_SITE_INVENTORY,
        {
            **_analysis_envelope("error_site_inventory"),
            "population_definition": (
                "SGV-XC1's oracle-localized OCR error sites that its region representation can "
                "express and SGV14's labeller can label: the set on which SGV-XC1's ground-truth "
                "corrector reached 1.0. Non-expressible and unlabelable sites are counted and not "
                "sampled."
            ),
            "sample_rule": {
                "per_environment": per_environment,
                "order": "sha256 of the stage seed and the site id",
                "seed": GEN1_SEED,
                "reads_a_label": False,
            },
            "environments": inventory,
            "totals": totals,
            "sampled_by_error_type": by_type,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"sites: {totals['evaluable_labelable']} evaluable of {totals['ocr_error_sites']} error "
        f"sites; {per_environment} per environment -> {totals['sampled']} sampled; "
        f"by type {by_type}"
    )
    return 0


# ------------------------------------------------------------------ sections 10, 37, 46-48


def _clean_spans(
    spec: dict[str, Any], page_pairs: Sequence[tuple[str, str]], pages: dict[tuple[str, str], Page]
) -> dict[tuple[str, str], tuple[set[str], list[tuple[int, int]]]]:
    """Per page: the OCR spans inside clean alignment components, and every error interval.

    Ground truth enters here, and only for the marker-shuffle control's decoy position. The decoy
    must be a place with no OCR error, which only the alignment can say.
    """
    alignment = pd.read_parquet(
        xc1.ALIGNMENT_SITES,
        columns=[
            "environment",
            "document_id",
            "site_kind",
            "evaluable",
            "d_before",
            "char_start",
            "char_end",
        ],
    )
    block = alignment[alignment["environment"] == spec["environment"]]
    out: dict[tuple[str, str], tuple[set[str], list[tuple[int, int]]]] = {}
    for pair in page_pairs:
        rows = block[block["document_id"] == pair[0]]
        clean = rows[(rows["site_kind"] == "clean") & rows["evaluable"] & (rows["d_before"] == 0)]
        errors = rows[~rows.index.isin(clean.index)]
        intervals = [
            (int(a), int(b)) for a, b in zip(errors["char_start"], errors["char_end"], strict=True)
        ]
        clean_ranges = [
            (int(a), int(b)) for a, b in zip(clean["char_start"], clean["char_end"], strict=True)
        ]
        page = pages[pair]
        spans = set()
        for span_id in page.order:
            s, e = page.span_start[span_id], page.span_end[span_id]
            inside = any(a <= s and e <= b for a, b in clean_ranges)
            touched = any(a < e and s < b for a, b in intervals)
            if inside and not touched:
                spans.add(span_id)
        out[pair] = (spans, intervals)
    return out


def _decoy(
    page: Page,
    anchors: Sequence[str],
    start: int,
    end: int,
    gap: bool,
    clean: set[str],
    errors: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """The nearest clean OCR position of the same marker kind, inside the site's E2 window."""
    window_start, window_end = line_window(page, anchors, neighbours=True)
    positions = [page.order.index(a) for a in anchors]
    centre = min(positions)
    options: list[tuple[int, int, int, int]] = []
    for index, span_id in enumerate(page.order):
        s, e = page.span_start[span_id], page.span_end[span_id]
        if s < window_start or e > window_end or span_id in anchors:
            continue
        if not gap:
            if span_id in clean:
                options.append((abs(index - centre), -index, s, e))
            continue
        if index + 1 >= len(page.order):
            continue
        right = page.order[index + 1]
        a, b = e, page.span_start[right]
        if right in anchors or b > window_end or (a, b) == (start, end):
            continue
        if span_id in clean and right in clean and not any(x <= b and a <= y for x, y in errors):
            options.append((abs(index - centre), -index, a, b))
    if not options:
        return None
    _distance, _index, a, b = min(options)
    return a, b


def _donor(sites: pd.DataFrame, label: str, name: str) -> dict[str, str]:
    """A seeded site-to-site mapping inside one environment that never stays on one document."""
    ordered = sites.sort_values("site_id", kind="stable").reset_index(drop=True)
    ids = ordered["site_id"].astype(str).tolist()
    documents = ordered["document_id"].astype(str).tolist()
    permutation = np.random.default_rng(_stable_seed(name, label)).permutation(len(ids)).tolist()
    mapping: dict[str, str] = {}
    for position, site in enumerate(ids):
        for step in range(len(ids)):
            candidate = permutation[(position + step) % len(ids)]
            if documents[candidate] != documents[position]:
                mapping[site] = ids[candidate]
                break
    return mapping


def run_contexts() -> int:
    """E0, E1, E2 and E3 for every sampled site, and the three control variants, frozen."""
    started = time.monotonic()
    _require(SAMPLE_SITES, "sites")
    for path in (CONTEXTS, CONTROL_CONTEXTS, CONTEXT_REGISTRY, CROP_REGISTRY):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("labels already exist; contexts are frozen before them")
    sample = pd.read_parquet(SAMPLE_SITES)
    frames: list[pd.DataFrame] = []
    controls: list[dict[str, Any]] = []
    census: list[dict[str, Any]] = []
    for spec in environment_specs():
        name = spec["environment"]
        began = time.monotonic()
        sites = sample[sample["environment"] == name].reset_index(drop=True)
        pages = _page_index(spec)
        contexts = build_contexts(sites, pages, CROP_DIR / _slug(name))
        frames.append(contexts)
        joined = sites.merge(contexts, on="site_id", validate="one_to_one")
        by_site = joined.set_index("site_id")
        pairs = sorted(
            {
                (str(d), str(e))
                for d, e in zip(sites["document_id"], sites["engine_id"], strict=True)
            }
        )
        clean = _clean_spans(spec, pairs, pages)
        subsample = joined[joined["in_control_subsample"]]
        context_donor = _donor(joined, X_CTX, name)
        image_donor = _donor(joined, X_IMG, name)
        counts = {
            "environment": name,
            "control_sites": len(subsample),
            X_MARK: 0,
            X_CTX: 0,
            X_IMG: 0,
        }
        for row in subsample.itertuples(index=False):
            site = str(row.site_id)
            pair = (str(row.document_id), str(row.engine_id))
            page = pages[pair]
            anchors = anchors_of(str(row.anchor_ref))
            gap = str(row.anchor_kind) == "gap"
            start, end = int(row.char_start), int(row.char_end)
            window_start, window_end = line_window(page, anchors, neighbours=True)
            window = page.stream[window_start:window_end]
            decoy = _decoy(page, anchors, start, end, gap, *clean[pair])
            if decoy is not None:
                a, b = decoy
                controls.append(
                    {
                        "site_id": site,
                        "condition": X_MARK,
                        "control_text": mark_text(window, a - window_start, b - window_start, gap),
                        "decoy_text": page.stream[a:b],
                        "donor_site_id": None,
                        "crop_path": None,
                        "crop_sha256": None,
                    }
                )
                counts[X_MARK] += 1
            donor = by_site.loc[context_donor[site]]
            donor_page = pages[str(donor["document_id"]), str(donor["engine_id"])]
            donor_anchors = anchors_of(str(donor["anchor_ref"]))
            d_start, d_end = line_window(donor_page, donor_anchors, neighbours=True)
            donor_window = donor_page.stream[d_start:d_end]
            ds, de = int(donor["char_start"]) - d_start, int(donor["char_end"]) - d_start
            own = mark_text(str(row.region), 0, len(str(row.region)), gap)
            controls.append(
                {
                    "site_id": site,
                    "condition": X_CTX,
                    "control_text": f"{donor_window[:ds]}{own}{donor_window[de:]}",
                    "decoy_text": None,
                    "donor_site_id": context_donor[site],
                    "crop_path": None,
                    "crop_sha256": None,
                }
            )
            counts[X_CTX] += 1
            image = by_site.loc[image_donor[site]]
            if image["crop_path"] is not None and not pd.isna(image["crop_path"]):
                controls.append(
                    {
                        "site_id": site,
                        "condition": X_IMG,
                        "control_text": str(row.e1_text),
                        "decoy_text": None,
                        "donor_site_id": image_donor[site],
                        "crop_path": str(image["crop_path"]),
                        "crop_sha256": str(image["crop_sha256"]),
                    }
                )
                counts[X_IMG] += 1
        census.append(
            {
                **counts,
                "sites": len(sites),
                "crops": int(contexts["crop_path"].notna().sum()),
                "e2_wider_than_e1": int(
                    (contexts["e2_text"].str.len() > contexts["e1_text"].str.len()).sum()
                ),
                "seconds": time.monotonic() - began,
            }
        )
        elapsed = time.monotonic() - began
        print(f"  {name}: {len(sites)} sites, {counts[X_MARK]} decoys ({elapsed:.0f}s)", flush=True)
    contexts_all = pd.concat(frames, ignore_index=True)
    control_frame = pd.DataFrame(controls)
    for frame, label in ((contexts_all, "contexts"), (control_frame, "control contexts")):
        leaked = [c for c in FORBIDDEN_INPUT_COLUMNS if c in frame.columns]
        if leaked:
            raise PhaseError(f"the {label} table carries truth columns: {leaked}")
    _write_parquet_once(CONTEXTS, contexts_all)
    _write_parquet_once(CONTROL_CONTEXTS, control_frame)
    lengths = {
        column: xc1._quantiles(contexts_all[column].str.len().tolist())
        for column in ("e0_text", "e1_text", "e2_text")
    }
    _write_json_once(
        CONTEXT_REGISTRY,
        {
            **_envelope("context_registry"),
            "conditions": cc_read_json(DESIGN_RECORD)["evidence_conditions"],
            "marker": {
                "span": f"{ERR_OPEN}<OCR text of the site>{ERR_CLOSE}",
                "gap": ERR_GAP,
                "gap_rendering": (
                    "the separator between the flanking tokens is replaced by the gap marker with "
                    "a space either side, or a space and a line break when the separator held one"
                ),
            },
            "policy": {
                "e0": f"{CONTEXT_CHARS} OCR characters each side, context line breaks flattened",
                "e1": "an exact slice of the OCR stream from the site's first to its last line",
                "e2": "the E1 slice widened to the previous and next line of the same page",
                "crosses_documents": False,
                "depends_on_outcome": False,
            },
            "controls": {
                X_MARK: (
                    "the E2 window with the marker moved to the nearest clean OCR position of the "
                    "same kind -- a clean token for a span site, a separator between clean tokens "
                    "for a gap site. The decoy is chosen with the alignment; the answer is scored "
                    "at the true site."
                ),
                X_CTX: (
                    "the E2 window of a site on another document of the same environment, with "
                    "that site's marked text replaced by this site's marked text"
                ),
                X_IMG: "this site's E1 text with the crop of a site on another document",
                "donor_rule": (
                    "a seeded permutation of the environment's sites, never the same document"
                ),
                "control_subsample_per_environment": CONTROL_PER_ENVIRONMENT,
            },
            "census": census,
            "text_length_characters": lengths,
            "contexts_table": {"path": _relative(CONTEXTS), "sha256": file_sha256(CONTEXTS)},
            "control_table": {
                "path": _relative(CONTROL_CONTEXTS),
                "sha256": file_sha256(CONTROL_CONTEXTS),
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        CROP_REGISTRY,
        {
            **_envelope("crop_registry"),
            "policy": {
                "box": (
                    "the union of the site's anchor-span OCR boxes and one OCR neighbour on each "
                    "side within the same line, padded by max("
                    f"{CROP_MIN_PAD} px, {CROP_PAD_FRACTION} x box height) and clamped to the page"
                ),
                "gap_sites": "the two flanking spans are the anchors, so the crop is their box",
                "resize": (
                    f"to height {CROP_HEIGHT} px with the aspect kept, width capped at "
                    f"{CROP_MAX_WIDTH} px, Lanczos"
                ),
                "uses_ground_truth": False,
                "frozen_before_endpoint_evaluation": True,
            },
            "image_availability": {
                "sites": len(contexts_all),
                "crops": int(contexts_all["crop_path"].notna().sum()),
                "unsupported_sites": int(contexts_all["crop_path"].isna().sum()),
            },
            "crops": {
                str(row.site_id): {
                    "box": row.crop_box.tolist()
                    if hasattr(row.crop_box, "tolist")
                    else row.crop_box,
                    "size": [row.crop_width, row.crop_height],
                    "sha256": row.crop_sha256,
                    "page_image_sha256": row.page_image_sha256,
                }
                for row in contexts_all.itertuples(index=False)
                if row.crop_sha256 is not None
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"contexts: {len(contexts_all)} sites, {int(contexts_all['crop_path'].notna().sum())} "
        f"crops, {len(control_frame)} control requests -> {_relative(CONTEXTS)}"
    )
    return 0


# ------------------------------------------------------------------ sections 8, 12, 15: registries


def run_registry() -> int:
    """The model, prompt and decoding registries, frozen together before any label exists."""
    started = time.monotonic()
    _require(CONTEXT_REGISTRY, "contexts")
    for path in (MODEL_REGISTRY, PROMPT_REGISTRY, DECODING_REGISTRY):
        _forbid(path)
    if _labels_exist():
        raise PhaseError("labels already exist; registries cannot be frozen after the fact")
    if not any(AUDIT_DIR.glob("*.json")):
        raise PhaseError("run --audit on a model first: the prompt is frozen on an audit")
    _write_json_once(
        MODEL_REGISTRY,
        {
            **_envelope("model_registry"),
            "gen1_models": {
                model: {
                    **{k: v for k, v in MODEL_SPECS[model].items() if k != "batch_size"},
                    "conditions": CELL_CONDITIONS[model],
                    "control_conditions": CONTROL_CONDITIONS_BY_MODEL[model],
                    "language_scope": ["eng", "deu"],
                    "analysis_only": True,
                    "oracle_localization": True,
                    "deployable": False,
                    "fine_tuned_here": False,
                }
                for model in GEN1_MODELS
            },
            "availability": (
                "each model's probe and format audit are written before it generates, and "
                f"{_relative(MODEL_AVAILABILITY)} assembles them before any label exists"
            ),
            "frozen_xc1_baselines": {
                M0: "SGV-XC1's frozen edit-aware generator on the oracle-located regions",
                M1: "SGV-XC1's ByT5 arm, ten beams, English scope only",
                M2: "SGV-XC1's Qwen2.5-1.5B arm under SGV-XC1's [[ ]] prompt, one sequence",
                "condition": XC1_ORACLE,
                "source": _relative(xc1.ORACLE_CANDIDATES),
            },
            "union": {
                M_UNION: "the deduplicated candidates of every GEN1 primary cell -- analysis only",
            },
            "frozen_before_endpoint_evaluation": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        PROMPT_REGISTRY,
        {
            **_envelope("prompt_registry"),
            "revision": PROMPT_REVISION,
            "system": SYSTEM_PROMPT,
            "user_templates": USER_TEMPLATES,
            "template_of_condition": TEMPLATE_OF,
            "output_contract": {
                "schema": {"candidates": ["<replacement>", "..."]},
                "max_candidates": MAX_CANDIDATES,
                "parse": "parse_candidates: strict JSON, else the first JSON object after a fence",
                "abstention": '{"candidates": []}',
                "deletion": 'the empty string "" at a span marker',
            },
            "shared_by_every_model": True,
            "contains_evaluation_examples": False,
            "contains_ground_truth": False,
            "revision_history": PROMPT_DRAFTS,
            "format_audits_at_freeze": {
                _relative(path): file_sha256(path) for path in sorted(AUDIT_DIR.glob("*.json"))
            },
            "prompt_hash": canonical_hash({"system": SYSTEM_PROMPT, "user": USER_TEMPLATES}),
            "frozen_before_endpoint_evaluation": True,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        DECODING_REGISTRY,
        {
            **_envelope("decoding_registry"),
            "shared": {
                "deterministic": True,
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "repetition_penalty": 1.0,
                "note": (
                    "every model decodes greedily by one rule. Sampling fields in a checkpoint's "
                    "generation_config are overridden, and so is any repetition penalty, which "
                    "would otherwise differ between checkpoints."
                ),
            },
            "revision": DECODING_REVISION,
            "max_new_tokens_rule": MAX_NEW_TOKENS_RULE,
            "cut_off_list": (
                "the candidate strings the decode completed are kept; nothing is guessed"
            ),
            "revision_history": [
                {
                    **entry,
                    "audit": _relative(SUPERSEDED_AUDIT),
                    "audit_sha256": (
                        file_sha256(SUPERSEDED_AUDIT) if SUPERSEDED_AUDIT.is_file() else None
                    ),
                }
                for entry in DECODING_HISTORY
            ],
            "batch_size": {m: int(MODEL_SPECS[m]["batch_size"]) for m in GEN1_MODELS},
            "ordering": "requests sorted by prompt length, ties by request id",
            "dtype": MODEL_DTYPE,
            "candidate_budget": {"max_candidates": MAX_CANDIDATES, "k_grid": list(K_GRID)},
            "reproducibility": (
                "a bf16 decode on this device repeats exactly for fixed batching and is not "
                "invariant to it; batch size and ordering are part of each arm's definition, and "
                "the audit records both properties"
            ),
            "decoding_hash": canonical_hash(
                {
                    "revision": DECODING_REVISION,
                    "rule": [GEN_MAX_NEW_TOKENS, GEN_BASE_TOKENS, GEN_PER_CANDIDATE_MARGIN],
                    "chars_per_token": CHARS_PER_TOKEN,
                    "gap": GAP_ALLOWANCE_CHARS,
                    "batch": {m: MODEL_SPECS[m]["batch_size"] for m in GEN1_MODELS},
                    "dtype": MODEL_DTYPE,
                }
            ),
            "searched_on_an_outcome": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"registry: {len(GEN1_MODELS)} models, prompt revision {PROMPT_REVISION} frozen")
    return 0


# ------------------------------------------------------------------ sections 40, 69, 70: generate


ALLOCATOR_VARIABLES = ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO")
# What happened to the Metal allocator on this 16 GiB machine, recorded because it decided how the
# runs were executed. A cap decides where memory lives and never what a kernel computes; M5's
# spot-check at the last cap regenerates answers first produced under the earlier one.
EXECUTION_INCIDENTS = (
    {
        "model": M3,
        "cap_ratio": None,
        "event": (
            "with no cap the format audit drove swap past 12 GB before any output existed; the "
            "run was stopped and nothing it produced was kept"
        ),
    },
    {
        "model": M3,
        "cap_ratio": 0.8,
        "event": (
            "Metal out of memory during the format audit at batch 16; relaunched at a cap of "
            "0.95, where every M3 shard was generated"
        ),
    },
    {
        "model": M5,
        "cap_ratio": 0.95,
        "event": (
            "Metal out of memory on the sbb/easyocr_de image shard after 16 shards, and again in "
            "a fresh process; the remaining 14 shards ran at a cap of 1.1, batch size unchanged"
        ),
    },
)


def _allocator_environment() -> dict[str, str | None]:
    """The Metal allocator limits in force. They decide where memory lives, never what a kernel
    computes, so they are execution settings recorded for provenance, not part of an arm."""
    return {name: os.environ.get(name) for name in ALLOCATOR_VARIABLES}


def _shard_path(model: str, condition: str, name: str) -> Path:
    return RAW_CACHE / f"{_slug(name)}.{model}.{condition}.parquet"


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name.replace(".parquet", ".cost.json"))


def _shard_complete(path: Path) -> bool:
    sidecar = _sidecar(path)
    if not (path.is_file() and sidecar.is_file()):
        return False
    return bool(cc_read_json(sidecar)["raw_output_sha256"] == file_sha256(path))


def _requests_for(model: str, condition: str, name: str) -> pd.DataFrame:
    """The frozen requests of one shard, rebuilt from the frozen GT-blind tables."""
    sample = pd.read_parquet(SAMPLE_SITES)
    sites = sample[sample["environment"] == name]
    if condition in CONTROL_CONDITIONS:
        sites = sites[sites["in_control_subsample"]]
        controls = pd.read_parquet(CONTROL_CONTEXTS)
        chosen = controls[controls["condition"] == condition][
            ["site_id", "control_text", "crop_path", "crop_sha256"]
        ]
        contexts = pd.read_parquet(CONTEXTS)[["site_id", "e0_text", "e1_text", "e2_text"]]
        chosen = chosen.merge(contexts, on="site_id", how="left")
    else:
        chosen = pd.read_parquet(CONTEXTS)
    return _cell_frame(sites, chosen, model, condition, "")


def run_generate() -> int:
    """Oracle-located, GT-blind inference, cached immutably one shard at a time.

    A shard is one (model, condition, environment). It is written atomically, with its cost and
    provenance sidecar beside it the moment it exists, so an interrupted run keeps everything it
    finished; a rerun skips every shard whose file matches its sidecar hash (section 70).
    """
    started = time.monotonic()
    _require(DECODING_REGISTRY, "registry")
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    decoding_hash = cc_read_json(DECODING_REGISTRY)["decoding_hash"]
    prompt_hash = cc_read_json(PROMPT_REGISTRY)["prompt_hash"]
    import transformers

    transformers.utils.logging.set_verbosity_error()
    device = _resolve_device()
    for model in _selected_models():
        spec = MODEL_SPECS[model]
        for directory, phase in ((PROBE_DIR, "probe"), (AUDIT_DIR, "audit")):
            if not (directory / f"{model}.json").is_file():
                raise PhaseError(f"{model}: run --{phase} before it generates (section 9)")
        work = [
            (condition, spec_env["environment"])
            for condition in (*CELL_CONDITIONS[model], *CONTROL_CONDITIONS_BY_MODEL[model])
            for spec_env in environment_specs()
            if not _shard_complete(_shard_path(model, condition, spec_env["environment"]))
        ]
        if not work:
            print(f"  {model}: every shard complete")
            continue
        began = time.monotonic()
        runner = _runner(model, device)
        print(f"  loaded {model} on {device} ({time.monotonic() - began:.0f}s); {len(work)} shards")
        for condition, name in work:
            path = _shard_path(model, condition, name)
            requests = _requests_for(model, condition, name)
            raw, cost = run_requests(runner, requests, int(spec["batch_size"]), print)
            temporary = path.with_name(path.name + ".partial")
            raw.to_parquet(temporary, index=False)
            temporary.rename(path)
            _sidecar(path).unlink(missing_ok=True)
            _write_json_once(
                _sidecar(path),
                {
                    **_analysis_envelope("generation_shard"),
                    "model": model,
                    "repo_id": spec["repo_id"],
                    "revision": spec["revision"],
                    "condition": condition,
                    "environment": name,
                    **cost,
                    "prompt_hash": prompt_hash,
                    "decoding_hash": decoding_hash,
                    "allocator_environment": _allocator_environment(),
                    "request_prompts_hash": canonical_hash(sorted(raw["prompt_sha256"].tolist())),
                    "device": device,
                    "raw_output": _relative(path),
                    "raw_output_sha256": file_sha256(path),
                    "clock": "time.monotonic, which does not advance while the machine sleeps",
                },
            )
            print(
                f"    {model}/{condition}/{name}: {cost['requests']} req in "
                f"{cost['wall_clock_seconds']:.0f}s ({cost['requests_per_second']:.2f}/s)",
                flush=True,
            )
        del runner
        _release_device_cache(device)
    print(f"generate: done in {time.monotonic() - started:.0f}s")
    return 0


REGENERATION_ENVIRONMENT = "funsd/doctr"
REGENERATION_BATCHES = 2


def run_spotcheck() -> int:
    """Section 73: each model regenerates its first batches, while its checkpoint is on disk."""
    REGENERATION_DIR.mkdir(parents=True, exist_ok=True)
    device = _resolve_device()
    for model in _selected_models():
        path = REGENERATION_DIR / f"{model}.json"
        if path.exists():
            continue
        condition = CELL_CONDITIONS[model][-1]
        shard = _shard_path(model, condition, REGENERATION_ENVIRONMENT)
        if not _shard_complete(shard) or _snapshot(model) is None:
            print(f"  {model}: shard or checkpoint missing; not spot-checked")
            continue
        runner = _runner(model, device)
        requests = _requests_for(model, condition, REGENERATION_ENVIRONMENT)
        batch = int(MODEL_SPECS[model]["batch_size"])
        again, _cost = run_requests(
            runner, requests, batch, lambda _m: None, limit_batches=REGENERATION_BATCHES
        )
        cached = pd.read_parquet(shard).set_index("request_id")["output"]
        differing = int(
            sum(cached[r] != o for r, o in zip(again["request_id"], again["output"], strict=True))
        )
        del runner
        _release_device_cache(device)
        _write_json_once(
            path,
            {
                **_envelope("model_regeneration"),
                "model": model,
                "condition": condition,
                "environment": REGENERATION_ENVIRONMENT,
                "batches": REGENERATION_BATCHES,
                "requests": len(again),
                "differing_outputs": differing,
                "identical": differing == 0,
            },
        )
        print(f"  {model}: {len(again)} regenerated, {differing} differ")
    return 0


# ------------------------------------------------------------------ the endpoint artifacts

GEN_CANDIDATES = OUT / "generation_candidates.parquet"
SITE_OUTCOMES = OUT / "site_outcomes.parquet"
XC1_RELABEL = CACHE / "xc1_m2_relabel.parquet"
RAW_OUTPUT_MANIFEST = OUT / "raw_output_manifest.json"
RAW_OUTPUT_HASHES = OUT / "raw_output_hashes.json"
NORMALIZED_CANDIDATES = OUT / "normalized_candidates.json"
FORMAT_COMPLIANCE = OUT / "format_compliance.json"
BASELINE_REPRODUCTION = OUT / "baseline_reproduction.json"
EXACT_RESULTS = OUT / "exact_generation_results.json"
TOPK_RESULTS = OUT / "topk_generation_results.json"
ERROR_TYPE_ANALYSIS = OUT / "error_type_analysis.json"
OMISSION_ANALYSIS = OUT / "omission_analysis.json"
SEGMENTATION_ANALYSIS = OUT / "segmentation_analysis.json"
SUBSTITUTION_ANALYSIS = OUT / "substitution_analysis.json"
ENGINE_ANALYSIS = OUT / "engine_analysis.json"
DOMAIN_ANALYSIS = OUT / "domain_analysis.json"
LANGUAGE_ANALYSIS = OUT / "language_typography_analysis.json"
CONTEXT_GAIN = OUT / "context_gain.json"
CAPACITY_GAIN = OUT / "capacity_gain.json"
VISUAL_GAIN = OUT / "visual_gain.json"
NEAR_MISS = OUT / "near_miss_analysis.json"
EDIT_DAMAGE = OUT / "edit_damage_analysis.json"
SPAN_ESCAPE = OUT / "span_escape_analysis.json"
MODEL_OVERLAP = OUT / "model_overlap.json"
UNIQUE_REPAIRS = OUT / "unique_repairs.json"
UNION_CEILING = OUT / "union_generation_ceiling.json"
RUNTIME_COST = OUT / "runtime_cost.json"
CONTROL_RESULTS = OUT / "control_results.json"
NEGATIVE_TESTS = OUT / "negative_tests.json"
STATISTICAL_TESTS = OUT / "statistical_tests.json"
DECISION = OUT / "research_decision.json"
DETERMINISM = OUT / "determinism.json"
PROVENANCE = OUT / "provenance.json"
TRACEABILITY = OUT / "traceability.json"
FIGURE_DIR = OUT / "figures"
FIGURE_MANIFEST = OUT / "figure_manifest.json"

# Sections 43-45. The controls that need no model run on every sampled site under one condition.
C_GT = "c_gt_corrector"
C_ID = "c_identity"
C_RAND = "c_random_replacement"
KNOWN_SITE = "known_site"
BASELINE_CELLS = tuple((m, XC1_ORACLE) for m in XC1_BASELINES)
MODEL_CONTROL_CELLS = tuple((m, c) for m, c, _r in CONTROL_CELLS)
PLAIN_CONTROL_CELLS = ((C_GT, KNOWN_SITE), (C_ID, KNOWN_SITE), (C_RAND, KNOWN_SITE))
RANDOM_ALPHABET = xc1.PERTURBATION_ALPHABET

LABEL_INPUT_COLUMNS = [
    "candidate_id",
    "site_id",
    "document_id",
    "engine_id",
    "original_ocr",
    "candidate_text",
    "char_start",
    "char_end",
]
SITE_META_COLUMNS = (
    "environment",
    "site_id",
    "document_id",
    "anchor_kind",
    "in_control_subsample",
)
# Character classes for section 22's substitution subtypes. Each is decidable from the OCR and the
# ground-truth region alone; none is a visual-similarity judgement made by hand.
HISTORICAL_CHARACTERS = frozenset("ſꝛͤ⸗ꝑꝙꝫ")


def cell_key(model: str, condition: str) -> str:
    return f"{model}::{condition}"


def _domain(corpus: str) -> str:
    return "modern_forms" if corpus == "funsd" else "historical_print"


def _label_requests(sites: pd.DataFrame, condition: str) -> pd.DataFrame:
    """SGV-XC1's request columns for one GEN1 condition, so its `_neural_edits` reads them."""
    digest = sites["site_id"].astype(str).str.split("|", n=1).str[1]
    frame = sites.copy()
    frame["condition"] = condition
    frame["request_id"] = (sites["environment"] + f"|{condition}|" + digest).to_numpy()
    return frame[list(REQUEST_COLUMNS)].reset_index(drop=True)


def _expand(
    raw: pd.DataFrame, requests: pd.DataFrame, model: str, condition: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Parsed answers in SGV-XC1's raw shape -- one row per candidate -- and one row per request."""
    by_id = requests.set_index("request_id")
    rows: list[dict[str, Any]] = []
    census: list[dict[str, Any]] = []
    for answer in raw.itertuples(index=False):
        request = by_id.loc[str(answer.request_id)]
        parsed = parse_candidates(str(answer.output), bool(answer.finished))
        flags = output_flags(str(answer.output))
        region = str(request["region"])
        gap = str(request["anchor_kind"]) == "gap"
        proposals = [c for c in parsed["candidates"] if c.strip() != region.strip()]
        valid = parsed["status"] in ("strict", "lenient")
        for rank, candidate in enumerate(parsed["candidates"]):
            rows.append(
                {
                    "environment": str(answer.environment),
                    "method": model,
                    "condition": condition,
                    "request_id": str(answer.request_id),
                    "rank": rank,
                    "output": candidate,
                }
            )
        census.append(
            {
                "site_id": str(request["site_id"]),
                "method": model,
                "condition": condition,
                "requested": True,
                "parse_status": str(parsed["status"]),
                "valid_output": valid,
                "listed_candidates": len(parsed["candidates"]),
                "proposals": len(proposals),
                "abstained": valid and not proposals,
                "empty_list": valid and not parsed["candidates"],
                "span_escapes": sum(
                    span_escape(
                        c,
                        region,
                        str(request["context_before"]),
                        str(request["context_after"]),
                        gap,
                    )
                    for c in proposals
                ),
                "echoed_prompt": flags["echoed_prompt"],
                "invalid_unicode": flags["invalid_unicode"],
                "over_limit": int(parsed["over_limit"]),
                "duplicates_removed": int(parsed["duplicates_removed"]),
                "non_string_items": int(parsed["non_string_items"]),
                "object_items": int(parsed["object_items"]),
                "finished": bool(answer.finished),
                "prompt_tokens": int(answer.prompt_tokens),
                "output_tokens": int(answer.output_tokens),
            }
        )
    return pd.DataFrame(rows, columns=list(RAW_COLUMNS)), census


def _random_outputs(requests: pd.DataFrame) -> pd.DataFrame:
    """Section 45: MAX_CANDIDATES seeded edits per site. Reads the OCR region and the seed only."""
    rows: list[dict[str, Any]] = []
    for row in requests.itertuples(index=False):
        rng = np.random.default_rng(_stable_seed(C_RAND, str(row.request_id)))
        gap = str(row.anchor_kind) == "gap"
        for rank in range(MAX_CANDIDATES):
            if gap:
                length = int(rng.integers(1, 4))
                text = "".join(
                    RANDOM_ALPHABET[int(rng.integers(len(RANDOM_ALPHABET)))] for _ in range(length)
                )
            else:
                text = xc1._perturb(str(row.region), rng)
            rows.append(
                {
                    "environment": str(row.environment),
                    "method": C_RAND,
                    "condition": KNOWN_SITE,
                    "request_id": str(row.request_id),
                    "rank": rank,
                    "output": text,
                }
            )
    return pd.DataFrame(rows, columns=list(RAW_COLUMNS))


def _model_cells() -> list[tuple[str, str]]:
    return [*PRIMARY_CELLS, *MODEL_CONTROL_CELLS]


def run_normalize() -> int:
    """Raw answers -> candidates -> SGV-XC1's atomic edits -> SGV14's labels, one call per page set.

    Every model cell and every control passes through the same parse, the same `_neural_edits`
    and the same `_labels` call, so a control is a test of the pipeline rather than of a separate
    path. SGV-XC1's own M2 raw outputs for the sampled sites are re-labelled through this path too,
    and the reproduction gate requires their labels to equal SGV-XC1's.
    """
    started = time.monotonic()
    for path in (
        GEN_CANDIDATES,
        SITE_OUTCOMES,
        XC1_RELABEL,
        NORMALIZED_CANDIDATES,
        FORMAT_COMPLIANCE,
        RAW_OUTPUT_MANIFEST,
        RAW_OUTPUT_HASHES,
    ):
        _forbid(path)
    for path in (MODEL_AVAILABILITY, DECODING_REGISTRY):
        _require(path, "availability" if path == MODEL_AVAILABILITY else "registry")
    missing = [
        _relative(_shard_path(m, c, spec["environment"]))
        for m, c in _model_cells()
        for spec in environment_specs()
        if not _shard_complete(_shard_path(m, c, spec["environment"]))
    ]
    if missing:
        raise PhaseError(f"{len(missing)} shards missing or unverified, e.g. {missing[:3]}")
    from ocr_risk.canonical import rebuild_stream

    sample = pd.read_parquet(SAMPLE_SITES)
    truth = pd.read_parquet(SAMPLE_TRUTH)
    xc1_candidates = pd.read_parquet(xc1.ORACLE_CANDIDATES)
    candidate_frames: list[pd.DataFrame] = []
    relabel_frames: list[pd.DataFrame] = []
    census_rows: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for spec in environment_specs():
        name = spec["environment"]
        began = time.monotonic()
        sites = sample[sample["environment"] == name].reset_index(drop=True)
        truth_env = truth[truth["environment"] == name].set_index("site_id")
        frames: list[tuple[str, str, pd.DataFrame]] = []
        for model, condition in _model_cells():
            path = _shard_path(model, condition, name)
            raw = pd.read_parquet(path)
            requests = _label_requests(sites, condition)
            expanded, census = _expand(raw, requests, model, condition)
            census_rows.extend(census)
            _sites, proposals, shard = xc1._neural_edits(
                name, cell_key(model, condition), expanded, requests, {}, None
            )
            frames.append((model, condition, proposals))
            shards.append(
                {
                    "environment": name,
                    "method": model,
                    "condition": condition,
                    **shard,
                    "proposals": len(proposals),
                }
            )
            manifest.append(
                {
                    "environment": name,
                    "method": model,
                    "condition": condition,
                    "raw_output": _relative(path),
                    "raw_output_sha256": file_sha256(path),
                    "sidecar": _relative(_sidecar(path)),
                    "requests": len(raw),
                }
            )
        known = _label_requests(sites, KNOWN_SITE)
        region_gt = known["site_id"].map(truth_env["region_gt"]).astype(str)
        plain = {
            C_GT: known.assign(output=region_gt.to_numpy()),
            C_ID: known.assign(output=known["region"].to_numpy()),
        }
        for method, frame in plain.items():
            raw = pd.DataFrame(
                {
                    "environment": name,
                    "method": method,
                    "condition": KNOWN_SITE,
                    "request_id": frame["request_id"].to_numpy(),
                    "rank": 0,
                    "output": frame["output"].to_numpy(),
                },
                columns=list(RAW_COLUMNS),
            )
            _sites, proposals, shard = xc1._neural_edits(
                name, cell_key(method, KNOWN_SITE), raw, known, {}, None
            )
            frames.append((method, KNOWN_SITE, proposals))
            shards.append({"environment": name, "method": method, "condition": KNOWN_SITE, **shard})
        _sites, proposals, shard = xc1._neural_edits(
            name, cell_key(C_RAND, KNOWN_SITE), _random_outputs(known), known, {}, None
        )
        frames.append((C_RAND, KNOWN_SITE, proposals))
        shards.append({"environment": name, "method": C_RAND, "condition": KNOWN_SITE, **shard})

        # SGV-XC1's M2 raw answers for the sampled sites, through this stage's own path.
        xc1_requests = pd.read_parquet(xc1.CACHE / f"{_slug(name)}.oracle_requests.parquet")
        xc1_requests = xc1_requests[xc1_requests["site_id"].isin(set(sites["site_id"]))]
        xc1_raw = pd.read_parquet(xc1.RAW_ORACLE_CACHE / f"{_slug(name)}.{M2}.{XC1_ORACLE}.parquet")
        xc1_raw = xc1_raw[xc1_raw["request_id"].isin(set(xc1_requests["request_id"]))]
        _sites, relabel, _shard = xc1._neural_edits(
            name, M2, xc1_raw, xc1_requests, {}, xc1.clean_llm_output
        )
        frames.append((M2, XC1_ORACLE, relabel))

        bundles = s14._document_bundles(spec["corpus"])
        roles = s14.partition_of(spec["corpus"], sorted(bundles))
        spans, _record = s14._canonical_spans(spec, bundles)
        streams = {pair: rebuild_stream(group) for pair, group in spans.items() if group}
        label_sites = sites[list(LABEL_SITE_COLUMNS)].assign(
            site_id=sites["site_id"].astype(str).str.split("|", n=1).str[1]
        )
        present = [(m, c, f) for m, c, f in frames if not f.empty]
        label_input = pd.concat(
            [f[LABEL_INPUT_COLUMNS] for _m, _c, f in present], ignore_index=True
        )
        labels, _diagnostics = s14._labels(
            spec["corpus"], bundles, spans, streams, label_sites, label_input
        )
        for method, condition, frame in present:
            labelled = xc1._method_frame(spec, method, condition, frame, labels, roles)
            if method == M2 and condition == XC1_ORACLE:
                relabel_frames.append(labelled)
                continue
            candidate_frames.append(labelled)
            shards.append(
                {
                    "environment": name,
                    "method": method,
                    "condition": condition,
                    "labelled_candidates": len(labelled),
                    "unlabelable_removed": len(frame) - len(labelled),
                }
            )
        baseline = xc1_candidates[
            xc1_candidates["method"].isin(XC1_BASELINES)
            & xc1_candidates["site_id"].isin(set(sites["site_id"]))
        ]
        candidate_frames.append(baseline)
        print(
            f"  {name}: {len(label_input)} proposals labelled ({time.monotonic() - began:.0f}s)",
            flush=True,
        )
    candidates = pd.concat(candidate_frames, ignore_index=True)
    census = pd.DataFrame(census_rows)
    outcomes = _site_outcomes(sample, truth, candidates, census)
    _write_parquet_once(GEN_CANDIDATES, candidates)
    _write_parquet_once(XC1_RELABEL, pd.concat(relabel_frames, ignore_index=True))
    _write_parquet_once(SITE_OUTCOMES, outcomes)
    _write_json_once(
        RAW_OUTPUT_MANIFEST,
        {**_analysis_envelope("raw_output_manifest"), "shards": manifest, "count": len(manifest)},
    )
    _write_json_once(
        RAW_OUTPUT_HASHES,
        {
            **_envelope("raw_output_hashes"),
            "hashes": {row["raw_output"]: row["raw_output_sha256"] for row in manifest},
            "verified_against_sidecars": True,
        },
    )
    _write_json_once(NORMALIZED_CANDIDATES, _normalization_record(shards, candidates, started))
    _write_json_once(FORMAT_COMPLIANCE, _format_record(census, started))
    print(f"normalize: {len(candidates)} labelled candidates, {len(outcomes)} site outcomes")
    return 0


def _site_outcomes(
    sample: pd.DataFrame, truth: pd.DataFrame, candidates: pd.DataFrame, census: pd.DataFrame
) -> pd.DataFrame:
    """One row per (cell, site in that cell's denominator): the table every analysis reads.

    A primary cell's denominator is the whole sample, a model control's is the control subsample.
    A site the cell never received a request for -- an E3 site without a crop, a decoy that could
    not be placed -- stays in the denominator as a missed repair, never as an absent row.
    """
    meta = sample[list(SITE_META_COLUMNS)].merge(
        truth[["site_id", "site_kind", "region_gt", "align_site_id"]], on="site_id"
    )
    meta = meta.merge(sample[["site_id", "region"]], on="site_id")
    by_env = {spec["environment"]: spec for spec in environment_specs()}
    meta["corpus"] = meta["environment"].map(lambda n: by_env[n]["corpus"])
    meta["base_engine"] = meta["environment"].map(lambda n: by_env[n]["base_engine"])
    meta["domain"] = meta["corpus"].map(_domain)
    meta["language"] = meta["corpus"].map(xc1.language_of)
    meta["typography"] = meta["environment"].map(lambda n: xc1._typography(by_env[n]))
    frame = candidates.copy()
    for k in K_GRID:
        frame[f"exact_at_{k}"] = frame["exact"] & (frame["generator_rank"] < k)
    grouped = frame.groupby(["method", "condition", "site_id"], sort=True)
    aggregate = grouped.agg(
        **{f"exact_at_{k}": (f"exact_at_{k}", "any") for k in K_GRID},
        exact_any=("exact", "any"),
        candidates=("candidate_id", "size"),
        harmful_candidates=("is_harmful", "sum"),
        beneficial_candidates=("beneficial", "sum"),
        exact_candidates=("exact", "sum"),
    ).reset_index()
    top = frame[frame["generator_rank"] == 0].drop_duplicates(["method", "condition", "site_id"])
    top = top[
        [
            "method",
            "condition",
            "site_id",
            "candidate_text",
            "outcome",
            "d_before",
            "d_after",
            "is_harmful",
            "exact",
        ]
    ].rename(
        columns={
            "candidate_text": "top1_text",
            "outcome": "top1_outcome",
            "d_before": "top1_d_before",
            "d_after": "top1_d_after",
            "is_harmful": "top1_harmful",
            "exact": "top1_exact",
        }
    )
    blocks: list[pd.DataFrame] = []
    cells = [*PRIMARY_CELLS, *MODEL_CONTROL_CELLS, *BASELINE_CELLS, *PLAIN_CONTROL_CELLS]
    for model, condition in cells:
        base = (
            meta[meta["in_control_subsample"]]
            if (model, condition) in MODEL_CONTROL_CELLS
            else meta
        )
        block = base.assign(method=model, condition=condition)
        block = block.merge(aggregate, on=["method", "condition", "site_id"], how="left")
        block = block.merge(top, on=["method", "condition", "site_id"], how="left")
        requested = census[(census["method"] == model) & (census["condition"] == condition)]
        block = block.merge(
            requested.drop(columns=["method", "condition"]), on="site_id", how="left"
        )
        if (model, condition) in BASELINE_CELLS:
            scope = {
                M0: block["anchor_kind"].isin(xc1.M0_ANCHOR_KINDS),
                M1: block["language"] == "eng",
                M2: pd.Series(True, index=block.index),
            }[model]
            block["requested"] = scope.to_numpy()
            block["parse_status"] = "not_applicable"
        elif (model, condition) in PLAIN_CONTROL_CELLS:
            block["requested"] = True
            block["parse_status"] = "not_applicable"
        blocks.append(block)
    outcomes = pd.concat(blocks, ignore_index=True)
    for column in [f"exact_at_{k}" for k in K_GRID] + ["exact_any", "requested", "top1_exact"]:
        outcomes[column] = outcomes[column].astype("boolean").fillna(False).astype(bool)
    for column in ("candidates", "harmful_candidates", "beneficial_candidates", "exact_candidates"):
        outcomes[column] = outcomes[column].fillna(0).astype(np.int64)
    for column in ("valid_output", "abstained", "empty_list"):
        outcomes[column] = outcomes[column].astype("boolean").fillna(False).astype(bool)
    outcomes["parse_status"] = outcomes["parse_status"].fillna("not_requested")
    return outcomes


def _normalization_record(
    shards: list[dict[str, Any]], candidates: pd.DataFrame, started: float
) -> dict[str, Any]:
    frame = pd.DataFrame(shards).fillna(0)
    numeric = [c for c in frame.columns if c not in ("environment", "method", "condition")]
    totals = frame.groupby(["method", "condition"])[numeric].sum().reset_index()
    inventory = (
        candidates.groupby(["method", "condition"])
        .agg(
            candidates=("candidate_id", "size"),
            sites=("site_id", "nunique"),
            exact=("exact", "sum"),
            harmful=("is_harmful", "sum"),
        )
        .reset_index()
    )
    return {
        **_analysis_envelope("normalized_candidates"),
        "path": (
            "raw answer -> parse_candidates -> one raw row per listed candidate -> SGV-XC1 "
            "_neural_edits (identity and duplicates dropped, rank kept) -> SGV14 _labels -> "
            "SGV-XC1 _method_frame"
        ),
        "by_cell": totals.to_dict("records"),
        "labelled_inventory": inventory.to_dict("records"),
        "runtime_seconds": time.monotonic() - started,
    }


def _format_record(census: pd.DataFrame, started: float) -> dict[str, Any]:
    """Section 35: every answer is accounted for; malformed ones are counted, never dropped."""
    records = []
    for (model, condition), block in census.groupby(["method", "condition"], sort=True):
        answers = len(block)
        status = block["parse_status"].value_counts().to_dict()
        records.append(
            {
                "method": model,
                "condition": condition,
                "answers": answers,
                **{
                    f"status_{k}": int(status.get(k, 0))
                    for k in ("strict", "lenient", "truncated", "invalid")
                },
                "valid_share": _ratio(int(block["valid_output"].sum()), answers),
                "unfinished_decodes": int((~block["finished"]).sum()),
                "empty_lists": int(block["empty_list"].sum()),
                "abstentions": int(block["abstained"].sum()),
                "abstention_share": _ratio(int(block["abstained"].sum()), answers),
                "echoed_prompt": int(block["echoed_prompt"].sum()),
                "invalid_unicode": int(block["invalid_unicode"].sum()),
                "over_limit_answers": int((block["over_limit"] > 0).sum()),
                "duplicates_removed": int(block["duplicates_removed"].sum()),
                "non_string_items": int(block["non_string_items"].sum()),
                "object_items": int(block["object_items"].sum()),
                "span_escape_candidates": int(block["span_escapes"].sum()),
                "proposals": int(block["proposals"].sum()),
                "max_prompt_tokens": int(block["prompt_tokens"].max()),
                "context_overflow": 0,
            }
        )
    return {
        **_analysis_envelope("format_compliance"),
        "cells": records,
        "context_overflow_note": (
            "no prompt approached any model's context window; the largest prompt per cell is "
            "recorded beside the window in model_availability_audit.json"
        ),
        "runtime_seconds": time.monotonic() - started,
    }


# ------------------------------------------------------------------ section 42: reproduction gate


def run_reproduce() -> int:
    """Section 42: SGV-XC1's known-site quantities, reproduced before any new model is read."""
    started = time.monotonic()
    _forbid(BASELINE_REPRODUCTION)
    _require(SITE_OUTCOMES, "normalize")
    checks: list[dict[str, Any]] = []
    stored = _xc1_json("oracle_localization")
    errors, arms = xc1.load_arms(xc1.ORACLE_CANDIDATES, xc1.ORACLE_LINKS)
    for method in XC1_BASELINES:
        observed = _mean([arms[name, method, XC1_ORACLE].opportunity_recall() for name in errors])
        expected = float(stored["per_method"][method]["mean_opportunity_recall"])
        checks.append(xc1._difference(f"{method}_known_site_opportunity", observed, expected))
    c3 = stored["c3_ground_truth_oracle_corrector"]
    inventory = cc_read_json(ERROR_SITE_INVENTORY)["totals"]
    for key, expected_key in (
        ("evaluable_labelable", "expressible_labelable_error_sites"),
        ("expressible", "expressible_error_sites"),
        ("ocr_error_sites", "ocr_error_sites"),
    ):
        checks.append(
            xc1._difference(f"population_{key}", int(inventory[key]), int(c3[expected_key]))
        )
    alignment = pd.read_parquet(xc1.ALIGNMENT_SITES)
    evaluation = alignment[
        (alignment["role"] == s14.ROLE_EVALUATION)
        & alignment["evaluable"]
        & (alignment["d_before"] > 0)
    ]
    counts = evaluation["site_kind"].value_counts().to_dict()
    frozen = _xc1_json("error_type_analysis")["by_condition"][xc1.PRIMARY_CONDITION][M0]
    for row in frozen:
        kind = str(row["stratum"])
        checks.append(
            xc1._difference(
                f"error_sites_{kind}", int(counts.get(kind, 0)), int(row["ocr_error_sites"])
            )
        )

    # This stage's site-level reading of each XC1 arm on the full labelable population, against
    # SGV-XC1's own linked reading restricted to the same source sites.
    truth = pd.concat(
        [
            pd.read_parquet(xc1.CACHE / f"{_slug(spec['environment'])}.oracle_truth.parquet")
            for spec in environment_specs()
        ],
        ignore_index=True,
    )
    table = pd.read_parquet(xc1.ORACLE_CANDIDATES)
    labelable = truth[
        truth["site_id"].isin(set(table.loc[table["method"] == C3_ORACLE, "site_id"]))
    ]
    align_of = dict(zip(labelable["site_id"], labelable["align_site_id"], strict=True))
    concordance: dict[str, Any] = {}
    for method in XC1_BASELINES:
        rows = table[(table["method"] == method) & table["exact"] & table["site_id"].isin(align_of)]
        site_level = {align_of[s] for s in rows["site_id"]}
        linked: set[str] = set()
        for name, population in errors.items():
            repaired = arms[name, method, XC1_ORACLE].repaired()
            linked |= set(population.ids[repaired].tolist())
        linked &= set(align_of.values())
        concordance[method] = {
            "site_level_repairs": len(site_level),
            "linked_repairs": len(linked),
            "only_site_level": len(site_level - linked),
            "only_linked": len(linked - site_level),
        }
        checks.append(
            xc1._difference(f"{method}_site_level_within_linked", len(site_level - linked), 0)
        )

    relabel = pd.read_parquet(XC1_RELABEL).set_index("candidate_id")
    original = table[(table["method"] == M2) & table["candidate_id"].isin(relabel.index)].set_index(
        "candidate_id"
    )
    checks.append(xc1._difference("xc1_m2_relabel_rows", len(relabel), len(original)))
    for column in ("outcome", "exact", "is_harmful", "beneficial", "d_before", "d_after"):
        differing = int((relabel.loc[original.index, column] != original[column]).sum())
        checks.append(xc1._difference(f"xc1_m2_relabel_{column}", differing, 0))

    outcomes = pd.read_parquet(SITE_OUTCOMES)
    gt = outcomes[(outcomes["method"] == C_GT) & (outcomes["condition"] == KNOWN_SITE)]
    checks.append(xc1._difference("gt_corrector_exact_sites", int(gt["exact_at_1"].sum()), len(gt)))
    sample_kinds = outcomes[(outcomes["method"] == C_GT) & (outcomes["condition"] == KNOWN_SITE)][
        "site_kind"
    ].value_counts()
    agrees = all(check["agrees"] for check in checks)
    _write_json_once(
        BASELINE_REPRODUCTION,
        {
            **_analysis_envelope("baseline_reproduction"),
            "checks": checks,
            "checks_total": len(checks),
            "checks_agreeing": sum(bool(c["agrees"]) for c in checks),
            "all_agree": agrees,
            "site_level_concordance": concordance,
            "concordance_note": (
                "this stage scores a known site by the candidates of its own request only. "
                "SGV-XC1's linking also credits an error site when a neighbouring request's "
                "candidate repairs it -- a gap insertion whose flanking span belongs to that "
                "site -- so the linked count may exceed the site-level count and never falls "
                "below it. "
                "The gate therefore requires site-level repairs to lie within linked repairs. A "
                "first draft required equality; a GT-blind dry run on fabricated answers found the "
                "difference in SGV-XC1's own M1 data, and the check was corrected to the invariant "
                "the two definitions guarantee before any GEN1 label existed."
            ),
            "sample_error_types": {str(k): int(v) for k, v in sample_kinds.items()},
            "expected_difference": 0,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"reproduce: {sum(c['agrees'] for c in checks)} of {len(checks)} checks agree")
    if not agrees:
        failed = [c["check"] for c in checks if not c["agrees"]]
        raise PhaseError(f"section 42 reproduction failed; do not interpret new models: {failed}")
    return 0


# ------------------------------------------------------------------ sections 17-18, 34: exact


def _outcomes() -> pd.DataFrame:
    _require(SITE_OUTCOMES, "normalize")
    return pd.read_parquet(SITE_OUTCOMES)


def _block(outcomes: pd.DataFrame, cell: tuple[str, str]) -> pd.DataFrame:
    return outcomes[(outcomes["method"] == cell[0]) & (outcomes["condition"] == cell[1])]


def _rates(
    outcomes: pd.DataFrame, cell: tuple[str, str], column: str = "exact_at_1"
) -> dict[str, Any]:
    block = _block(outcomes, cell)
    per_env = block.groupby("environment", sort=True)[column].mean()
    return {
        "per_environment": {str(k): float(v) for k, v in per_env.items()},
        "mean": float(per_env.mean()) if len(per_env) else float("nan"),
        "pooled": float(block[column].mean()) if len(block) else float("nan"),
        "sites": len(block),
        "repaired_sites": int(block[column].sum()),
    }


def _paired(
    outcomes: pd.DataFrame,
    a: tuple[str, str],
    b: tuple[str, str],
    column: str = "exact_at_1",
    mask: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    """The two cells on the sites they share, one row per site."""
    left = _block(outcomes, a).set_index("site_id")
    right = _block(outcomes, b).set_index("site_id")
    shared = left.index.intersection(right.index)
    frame = left.loc[
        shared, ["environment", "document_id", "site_kind", "anchor_kind", "domain"]
    ].copy()
    frame["a"] = left.loc[shared, column].astype(float)
    frame["b"] = right.loc[shared, column].astype(float)
    if mask is not None:
        frame = frame[mask(frame)]
    return frame


def effect(
    outcomes: pd.DataFrame, a: tuple[str, str], b: tuple[str, str], column: str = "exact_at_1"
) -> dict[str, Any]:
    """Sections 27-29 and 52-54: one effect, and whether it is material by the frozen rule."""
    frame = _paired(outcomes, a, b, column)
    per_env = (frame["a"] - frame["b"]).groupby(frame["environment"], sort=True).mean()
    by_type = {
        str(kind): float((block["a"] - block["b"]).mean())
        for kind, block in frame.groupby("site_kind", sort=True)
    }
    by_type_sites = {str(kind): len(block) for kind, block in frame.groupby("site_kind")}
    meaningful = [str(env) for env, value in per_env.items() if value >= MEANINGFUL_DELTA]
    weak = [kind for kind in WEAK_ERROR_TYPES if by_type.get(kind, 0.0) >= WEAK_TYPE_DELTA]
    mean = float(per_env.mean())
    return {
        "a": cell_key(*a),
        "b": cell_key(*b),
        "metric": column,
        "sites": len(frame),
        "a_mean": float(frame.groupby("environment")["a"].mean().mean()),
        "b_mean": float(frame.groupby("environment")["b"].mean().mean()),
        "mean_delta": mean,
        "pooled_delta": float((frame["a"] - frame["b"]).mean()),
        "per_environment_delta": {str(k): float(v) for k, v in per_env.items()},
        "environments_improved": int((per_env > 0).sum()),
        "environments_worsened": int((per_env < 0).sum()),
        "environments_meaningfully_improved": len(meaningful),
        "environments_meaningfully_improved_names": meaningful,
        "delta_by_error_type": by_type,
        "sites_by_error_type": by_type_sites,
        "weak_types_improved": weak,
        "meets_margin": mean >= MEANINGFUL_DELTA,
        "meets_breadth": len(meaningful) >= BREADTH_MAJORITY,
        "meets_weak_types": len(weak) >= WEAK_TYPES_REQUIRED,
        "material": bool(
            mean >= MEANINGFUL_DELTA
            and len(meaningful) >= BREADTH_MAJORITY
            and len(weak) >= WEAK_TYPES_REQUIRED
        ),
    }


def _cell_summary(outcomes: pd.DataFrame, cell: tuple[str, str]) -> dict[str, Any]:
    block = _block(outcomes, cell)
    rates = {f"exact_at_{k}": _rates(outcomes, cell, f"exact_at_{k}") for k in K_GRID}
    requested = block[block["requested"]]
    candidates = int(block["candidates"].sum())
    return {
        "method": cell[0],
        "condition": cell[1],
        "cell": cell_key(*cell),
        "eligible_sites": len(block),
        "in_scope_sites": int(block["requested"].sum()),
        "model_requests": int(block["requested"].sum()),
        "valid_outputs": int(block["valid_output"].sum()),
        "invalid_outputs": int(requested["parse_status"].isin(["invalid", "truncated"]).sum()),
        "abstentions": int(block["abstained"].sum()),
        "abstention_rate": _ratio(int(block["abstained"].sum()), len(block)),
        "exact_top1": int(block["exact_at_1"].sum()),
        "exact_generation": rates["exact_at_1"]["mean"],
        "exact_generation_pooled": rates["exact_at_1"]["pooled"],
        "exact_at_k": {str(k): rates[f"exact_at_{k}"]["mean"] for k in K_GRID},
        "exact_at_k_pooled": {str(k): rates[f"exact_at_{k}"]["pooled"] for k in K_GRID},
        "exact_any_candidate": _rates(outcomes, cell, "exact_any")["mean"],
        "per_environment_exact_at_1": rates["exact_at_1"]["per_environment"],
        "candidates": candidates,
        "candidates_per_site": _ratio(candidates, len(block)),
        "exact_candidates": int(block["exact_candidates"].sum()),
        "harmful_candidates": int(block["harmful_candidates"].sum()),
        "candidate_precision": _ratio(int(block["exact_candidates"].sum()), candidates),
        "harmful_share_of_candidates": _ratio(int(block["harmful_candidates"].sum()), candidates),
    }


def run_exact() -> int:
    """Sections 17, 18 and 34: ExactGeneration@K for every cell, with explicit denominators."""
    started = time.monotonic()
    for path in (EXACT_RESULTS, TOPK_RESULTS):
        _forbid(path)
    outcomes = _outcomes()
    cells = [*PRIMARY_CELLS, *BASELINE_CELLS]
    summaries = {cell_key(*cell): _cell_summary(outcomes, cell) for cell in cells}
    reference = (M2G, E1)
    for cell in cells:
        record = summaries[cell_key(*cell)]
        record["delta_vs_m2g_e1"] = (
            record["exact_generation"] - summaries[cell_key(*reference)]["exact_generation"]
        )
        record["delta_vs_m0"] = (
            record["exact_generation"] - summaries[cell_key(M0, XC1_ORACLE)]["exact_generation"]
        )
    primary = [cell_key(*c) for c in PRIMARY_CELLS]
    best = max(primary, key=lambda k: summaries[k]["exact_generation"])
    _write_json_once(
        EXACT_RESULTS,
        {
            **_analysis_envelope("exact_generation_results"),
            "primary_metric": "exact_generation_at_1, mean over environments of a site rate",
            "denominator": "the sampled evaluable known error sites of the cell's environment",
            "cells": summaries,
            "gen1_cells": primary,
            "baseline_cells": [cell_key(*c) for c in BASELINE_CELLS],
            "best_gen1_cell": best,
            "best_gen1_exact_generation": summaries[best]["exact_generation"],
            "reference_cell": cell_key(*reference),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    curves = {
        key: {
            "exact_at_k": record["exact_at_k"],
            "monotone_in_k": all(
                record["exact_at_k"][str(a)] <= record["exact_at_k"][str(b)]
                for a, b in pairwise(K_GRID)
            ),
            "top5_minus_top1": record["exact_at_k"]["5"] - record["exact_at_k"]["1"],
        }
        for key, record in summaries.items()
    }
    _write_json_once(
        TOPK_RESULTS,
        {
            **_analysis_envelope("topk_generation_results"),
            "k_grid": list(K_GRID),
            "note": (
                "section 55: K=3 and K=5 are prefixes of the same ranked list and measure ranking "
                "headroom -- whether the model produces the answer at all -- never solved "
                "generation"
            ),
            "cells": curves,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        "exact: "
        + ", ".join(f"{k} {summaries[k]['exact_generation']:.4f}" for k in primary)
        + f"; best {best}"
    )
    return 0


# ------------------------------------------------------------------ sections 19-22, 30-32: strata


def substitution_subtype(ocr: str, gt: str) -> str:
    """Section 22, decided from the two strings alone, in a fixed order."""
    from rapidfuzz.distance import Levenshtein

    if ocr.lower() == gt.lower() and ocr != gt:
        return "case_only"
    operations = Levenshtein.editops(ocr, gt)
    pairs = [(ocr[s], gt[d]) for tag, s, d in operations if tag == "replace"]
    touched = [c for pair in pairs for c in pair]
    touched += [ocr[s] for tag, s, _d in operations if tag == "delete"]
    touched += [gt[d] for tag, _s, d in operations if tag == "insert"]
    if any(c in HISTORICAL_CHARACTERS or unicodedata.category(c) == "Mn" for c in touched):
        return "historical_glyph"
    if pairs and all(a.isalnum() and b.isalnum() and a.isdigit() != b.isdigit() for a, b in pairs):
        return "digit_letter"
    if any(unicodedata.category(c)[0] in "PS" for c in touched):
        return "punctuation_or_symbol"
    if any(ord(c) > 127 for c in touched):
        return "other_non_ascii"
    if pairs and all(a.isalpha() and b.isalpha() for a, b in pairs):
        return "letter_letter"
    return "other"


def segmentation_subtype(ocr: str, gt: str) -> str:
    """Section 21: whitespace-only boundary errors, split by direction, or mixed with characters."""
    if "".join(ocr.split()) != "".join(gt.split()):
        return "boundary_with_character_errors"
    ocr_words, gt_words = len(ocr.split()), len(gt.split())
    if gt_words > ocr_words:
        return "missing_space_merge"
    if gt_words < ocr_words:
        return "spurious_space_split"
    return "boundary_placement"


def _strata_rows(
    outcomes: pd.DataFrame, cells: Sequence[tuple[str, str]], key: str
) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        block = _block(outcomes, cell)
        for value, group in block.groupby(key, sort=True):
            rows.append(
                {
                    "cell": cell_key(*cell),
                    "method": cell[0],
                    "condition": cell[1],
                    key: str(value),
                    "sites": len(group),
                    "exact_at_1": float(group["exact_at_1"].mean()),
                    "exact_at_5": float(group["exact_at_5"].mean()),
                    "abstention_rate": float(group["abstained"].mean()),
                    "repaired_at_1": int(group["exact_at_1"].sum()),
                }
            )
    return rows


def run_strata() -> int:
    """Sections 19-22 and 30-32: error type and its subtypes, engine, domain and typography."""
    started = time.monotonic()
    for path in (
        ERROR_TYPE_ANALYSIS,
        OMISSION_ANALYSIS,
        SEGMENTATION_ANALYSIS,
        SUBSTITUTION_ANALYSIS,
        ENGINE_ANALYSIS,
        DOMAIN_ANALYSIS,
        LANGUAGE_ANALYSIS,
    ):
        _forbid(path)
    from ocr_risk.metrics.text import levenshtein

    outcomes = _outcomes()
    cells = [*PRIMARY_CELLS, *BASELINE_CELLS]
    _write_json_once(
        ERROR_TYPE_ANALYSIS,
        {
            **_analysis_envelope("error_type_analysis"),
            "taxonomy": {
                "deletion": "OCR omission: text missing from the OCR (a gap site)",
                "insertion": "spurious insertion: OCR text with no ground truth",
                "segmentation": "a word-boundary error",
                "substitution": "wrong characters in place of right ones",
            },
            "weak_error_types": list(WEAK_ERROR_TYPES),
            "strong_error_type": STRONG_ERROR_TYPE,
            "rows": _strata_rows(outcomes, cells, "site_kind"),
            "runtime_seconds": time.monotonic() - started,
        },
    )

    omission = outcomes[outcomes["site_kind"] == "deletion"].copy()
    omission["gt_length"] = omission["region_gt"].astype(str).str.len()
    omission_cells = []
    for cell in cells:
        block = _block(omission, cell)
        wrong = block[block["top1_text"].notna() & ~block["top1_exact"]]
        distances = [
            levenshtein(str(t), str(g))
            for t, g in zip(wrong["top1_text"], wrong["region_gt"], strict=True)
        ]
        omission_cells.append(
            {
                "cell": cell_key(*cell),
                "sites": len(block),
                "exact_at_1": float(block["exact_at_1"].mean()),
                "exact_at_5": float(block["exact_at_5"].mean()),
                "abstention_rate": float(block["abstained"].mean()),
                "wrong_top1": len(wrong),
                "wrong_top1_distance": xc1._quantiles(distances),
                "wrong_top1_inserted_length": xc1._quantiles(
                    wrong["top1_text"].astype(str).str.len().tolist()
                ),
                "exact_by_gt_length": {
                    label: float(group["exact_at_1"].mean())
                    for label, group in block.groupby(
                        pd.cut(
                            block["gt_length"],
                            [0, 1, 3, 10, 10**6],
                            labels=["1", "2-3", "4-10", ">10"],
                        ),
                        observed=True,
                    )
                },
            }
        )
    _write_json_once(
        OMISSION_ANALYSIS,
        {
            **_analysis_envelope("omission_analysis"),
            "gt_insertion_length": xc1._quantiles(
                _block(omission, (C_GT, KNOWN_SITE))["gt_length"].tolist()
            ),
            "cells": omission_cells,
            "visual_contrast": effect(omission, (M5, E3), (M5, E1))
            if len(_block(omission, (M5, E3)))
            else None,
            "image_support_note": (
                "whether an individual generated insertion is supported by the image cannot be "
                "judged without a second reader; the matched text-only versus image contrast on "
                "omission sites is reported instead"
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )

    segmentation = outcomes[outcomes["site_kind"] == "segmentation"].copy()
    segmentation["subtype"] = [
        segmentation_subtype(str(o), str(g))
        for o, g in zip(segmentation["region"], segmentation["region_gt"], strict=True)
    ]
    _write_json_once(
        SEGMENTATION_ANALYSIS,
        {
            **_analysis_envelope("segmentation_analysis"),
            "subtypes": {
                "missing_space_merge": "the OCR joined words the ground truth separates",
                "spurious_space_split": "the OCR split a word the ground truth keeps whole",
                "boundary_placement": "the same number of words, boundaries in other places",
                "boundary_with_character_errors": "a boundary error with character errors too",
            },
            "sample_counts": {
                str(k): int(v)
                for k, v in _block(segmentation, (C_GT, KNOWN_SITE))["subtype"]
                .value_counts()
                .items()
            },
            "rows": _strata_rows(segmentation, cells, "subtype"),
            "runtime_seconds": time.monotonic() - started,
        },
    )

    substitution = outcomes[outcomes["site_kind"] == "substitution"].copy()
    substitution["subtype"] = [
        substitution_subtype(str(o), str(g))
        for o, g in zip(substitution["region"], substitution["region_gt"], strict=True)
    ]
    _write_json_once(
        SUBSTITUTION_ANALYSIS,
        {
            **_analysis_envelope("substitution_analysis"),
            "subtypes": {
                "case_only": "the strings differ only in letter case",
                "historical_glyph": "an edited character is a historical letter form or mark",
                "digit_letter": "every substituted pair crosses digit and letter",
                "punctuation_or_symbol": "an edited character is punctuation or a symbol",
                "other_non_ascii": "an edited character is another non-ASCII character",
                "letter_letter": "every substituted pair is letter for letter",
                "other": "none of the above",
            },
            "order": "the first matching subtype in the order listed",
            "sample_counts": {
                str(k): int(v)
                for k, v in _block(substitution, (C_GT, KNOWN_SITE))["subtype"]
                .value_counts()
                .items()
            },
            "rows": _strata_rows(substitution, cells, "subtype"),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    for path, key, name in (
        (ENGINE_ANALYSIS, "environment", "engine_analysis"),
        (DOMAIN_ANALYSIS, "domain", "domain_analysis"),
        (LANGUAGE_ANALYSIS, "typography", "language_typography_analysis"),
    ):
        rows = _strata_rows(outcomes, cells, key)
        extra: dict[str, Any] = {}
        if key == "environment":
            extra["by_base_engine"] = _strata_rows(outcomes, cells, "base_engine")
        if key == "typography":
            extra["by_language"] = _strata_rows(outcomes, cells, "language")
        _write_json_once(
            path,
            {
                **_analysis_envelope(name),
                "rows": rows,
                **extra,
                "runtime_seconds": time.monotonic() - started,
            },
        )
    print(f"strata: {len(cells)} cells by error type, subtype, engine, domain and typography")
    return 0


# ------------------------------------------------------------------ sections 27-29: gains


def run_gains() -> int:
    """Sections 27-29: the capacity, context and visual effects, each judged by the frozen rule."""
    started = time.monotonic()
    for path in (CAPACITY_GAIN, CONTEXT_GAIN, VISUAL_GAIN):
        _forbid(path)
    outcomes = _outcomes()

    def effects(specs: Sequence[tuple[str, tuple[str, str], tuple[str, str]]]) -> dict[str, Any]:
        return {
            name: {
                **effect(outcomes, a, b),
                "at_k": {
                    str(k): effect(outcomes, a, b, f"exact_at_{k}")["mean_delta"] for k in K_GRID
                },
            }
            for name, a, b in specs
        }

    capacity = effects(CAPACITY_EFFECTS)
    joint = effects(JOINT_EFFECTS)
    context = effects([e for e in EVIDENCE_EFFECTS if not e[0].startswith("visual")])
    visual = effects([e for e in EVIDENCE_EFFECTS if e[0].startswith("visual")])
    confound = (
        "the visual effect is measured on one set of weights with and without the crop, so it "
        "is not confounded with model size; the capacity effects compare different models at "
        "one evidence level, so they are not confounded with evidence"
    )
    _write_json_once(
        CAPACITY_GAIN,
        {
            **_analysis_envelope("capacity_gain"),
            "definition": "ExactGeneration@1(strong model, E) - ExactGeneration@1(M2G, E)",
            "effects": capacity,
            "joint_effects": joint,
            "any_material": any(r["material"] for r in capacity.values()),
            "any_joint_material": any(r["material"] for r in joint.values()),
            "confounding": confound,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        CONTEXT_GAIN,
        {
            **_analysis_envelope("context_gain"),
            "definition": "ExactGeneration@1(model, richer E) - ExactGeneration@1(model, poorer E)",
            "effects": context,
            "any_material": any(r["material"] for r in context.values()),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        VISUAL_GAIN,
        {
            **_analysis_envelope("visual_gain"),
            "definition": "ExactGeneration@1(M5, E3) - ExactGeneration@1(M5, E1): same weights",
            "effects": visual,
            "any_material": any(r["material"] for r in visual.values()),
            "confounding": confound,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        "gains: capacity "
        + ", ".join(f"{k} {v['mean_delta']:+.4f}" for k, v in capacity.items())
        + "; context "
        + ", ".join(f"{k} {v['mean_delta']:+.4f}" for k, v in context.items())
        + "; visual "
        + ", ".join(f"{k} {v['mean_delta']:+.4f}" for k, v in visual.items())
    )
    return 0


# ------------------------------------------------------------------ sections 24-26, 36: diagnostics


def _analysis_cells() -> list[tuple[str, str]]:
    return [*PRIMARY_CELLS, *BASELINE_CELLS]


def run_diagnostics() -> int:
    """Near misses, wrong-edit damage and span escape. Secondary: none enters the endpoint."""
    started = time.monotonic()
    for path in (NEAR_MISS, EDIT_DAMAGE, SPAN_ESCAPE):
        _forbid(path)
    from ocr_risk.metrics.text import cer, levenshtein

    outcomes = _outcomes()
    candidates = pd.read_parquet(GEN_CANDIDATES)
    truth = pd.read_parquet(SAMPLE_TRUTH).set_index("site_id")["region_gt"]
    near: dict[str, Any] = {}
    damage: dict[str, Any] = {}
    escape: dict[str, Any] = {}
    for cell in _analysis_cells():
        block = _block(outcomes, cell)
        wrong = block[block["top1_text"].notna() & ~block["top1_exact"]]
        pairs = list(
            zip(wrong["top1_text"].astype(str), wrong["region_gt"].astype(str), strict=True)
        )
        distances = [levenshtein(t, g) for t, g in pairs]
        normalized = [d / max(1, len(g)) for d, (_t, g) in zip(distances, pairs, strict=True)]
        rates = [cer(g, t) for t, g in pairs if g]
        prefix = [len(os.path.commonprefix([t, g])) / max(1, len(g)) for t, g in pairs]
        suffix = [len(os.path.commonprefix([t[::-1], g[::-1]])) / max(1, len(g)) for t, g in pairs]
        near[cell_key(*cell)] = {
            "wrong_top1_candidates": len(pairs),
            "edit_distance_to_gt": xc1._quantiles(distances),
            "normalized_edit_distance": xc1._quantiles(normalized),
            "candidate_cer_against_gt": xc1._quantiles(rates),
            "share_within_one_character": _ratio(sum(d == 1 for d in distances), len(distances)),
            "share_within_two_characters": _ratio(sum(d <= 2 for d in distances), len(distances)),
            "common_prefix_share": xc1._quantiles(prefix),
            "common_suffix_share": xc1._quantiles(suffix),
            "ocr_distance_before": xc1._quantiles(wrong["top1_d_before"].dropna().tolist()),
        }
        rows = candidates[(candidates["method"] == cell[0]) & (candidates["condition"] == cell[1])]
        wrong_rows = rows[~rows["exact"]]
        top = wrong_rows[wrong_rows["generator_rank"] == 0]
        damage[cell_key(*cell)] = {
            **{f"all_{key}": value for key, value in _damage(wrong_rows, len(rows)).items()},
            **{
                f"top1_{key}": value
                for key, value in _damage(top, int((rows["generator_rank"] == 0).sum())).items()
            },
        }
        answered = block[block["requested"] & block["valid_output"]]
        proposals = int(block["proposals"].fillna(0).sum())
        escapes = int(block["span_escapes"].fillna(0).sum())
        escape[cell_key(*cell)] = {
            "proposals": proposals,
            "span_escape_candidates": escapes,
            "span_escape_rate": _ratio(escapes, proposals),
            "answers_with_an_escape": int((block["span_escapes"].fillna(0) > 0).sum()),
            "valid_answers": len(answered),
            "applicable": cell not in BASELINE_CELLS,
        }
    common = {"truth_rows": len(truth)}
    _write_json_once(
        NEAR_MISS,
        {
            **_analysis_envelope("near_miss_analysis"),
            "unit": "the top-1 candidate at a site where it is not an exact repair",
            "distance": "ocr_risk.metrics.text.levenshtein against the region ground truth",
            "cer": "ocr_risk.metrics.text.cer, where the region ground truth is non-empty",
            "cells": near,
            **common,
            "note": "section 24: a near miss is never counted as a repair",
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        EDIT_DAMAGE,
        {
            **_analysis_envelope("edit_damage_analysis"),
            "classes": {
                "improves_not_exact": "label d_after < d_before and not an exact repair",
                "equivalent_distance": "label d_after == d_before",
                "worsens": "label d_after > d_before",
                "harmful": "the frozen STRICT_WORSENING label, unchanged",
            },
            "cells": damage,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    _write_json_once(
        SPAN_ESCAPE,
        {
            **_analysis_envelope("span_escape_analysis"),
            "definition": (
                "a proposed candidate that carries a marker, a line break the region lacks, or "
                "the whitespace-delimited OCR word on either side of the site. Candidates are "
                "scored as produced: an escaped candidate is not repaired into an exact one."
            ),
            "cells": escape,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"diagnostics: {len(near)} cells")
    return 0


def _damage(rows: pd.DataFrame, total: int) -> dict[str, Any]:
    improves = int((rows["d_after"] < rows["d_before"]).sum())
    equal = int((rows["d_after"] == rows["d_before"]).sum())
    worsens = int((rows["d_after"] > rows["d_before"]).sum())
    return {
        "candidates": total,
        "wrong_candidates": len(rows),
        "improves_not_exact": improves,
        "equivalent_distance": equal,
        "worsens": worsens,
        "harmful": int(rows["is_harmful"].sum()),
        "worsens_share_of_wrong": _ratio(worsens, len(rows)),
        "improves_share_of_wrong": _ratio(improves, len(rows)),
    }


# ------------------------------------------------------------------ sections 58-59: overlap


def run_overlap() -> int:
    """Section 58's complementarity and section 59's union ceiling. Analysis only."""
    started = time.monotonic()
    for path in (MODEL_OVERLAP, UNIQUE_REPAIRS, UNION_CEILING):
        _forbid(path)
    outcomes = _outcomes()
    candidates = pd.read_parquet(GEN_CANDIDATES)
    cells = list(PRIMARY_CELLS)
    sites = _block(outcomes, (C_GT, KNOWN_SITE))[["site_id", "environment", "site_kind"]]

    def repaired(cell: tuple[str, str], column: str) -> set[str]:
        block = _block(outcomes, cell)
        return set(block.loc[block[column], "site_id"])

    pairs: list[dict[str, Any]] = []
    for column in ("exact_at_1", "exact_at_5"):
        sets = {cell_key(*c): repaired(c, column) for c in [*cells, (M0, XC1_ORACLE)]}
        keys = list(sets)
        for i, left in enumerate(keys):
            for right in keys[i + 1 :]:
                both = sets[left] & sets[right]
                either = sets[left] | sets[right]
                pairs.append(
                    {
                        "metric": column,
                        "a": left,
                        "b": right,
                        "both": len(both),
                        "only_a": len(sets[left] - sets[right]),
                        "only_b": len(sets[right] - sets[left]),
                        "jaccard": _ratio(len(both), len(either)),
                    }
                )
    _write_json_once(
        MODEL_OVERLAP,
        {
            **_analysis_envelope("model_overlap"),
            "pairs": pairs,
            "runtime_seconds": time.monotonic() - started,
        },
    )

    by_model: dict[str, set[str]] = {}
    for model in GEN1_MODELS:
        for condition in CELL_CONDITIONS[model]:
            by_model.setdefault(model, set()).update(repaired((model, condition), "exact_at_5"))
    by_model[M0] = repaired((M0, XC1_ORACLE), "exact_at_5")
    kind = dict(zip(sites["site_id"], sites["site_kind"], strict=True))
    unique: dict[str, Any] = {}
    for model, found in by_model.items():
        others = set().union(*(s for m, s in by_model.items() if m != model))
        only = found - others
        unique[model] = {
            "repaired_sites_top5": len(found),
            "unique_repairs": len(only),
            "unique_by_error_type": {
                str(k): v for k, v in pd.Series([kind[s] for s in only]).value_counts().items()
            }
            if only
            else {},
        }
    _write_json_once(
        UNIQUE_REPAIRS,
        {
            **_analysis_envelope("unique_repairs"),
            "unit": (
                "sites repaired within the top 5 by any condition of the model, and by no other"
            ),
            "models": unique,
            "runtime_seconds": time.monotonic() - started,
        },
    )

    def union_rate(chosen: Sequence[tuple[str, str]], column: str) -> dict[str, Any]:
        found: set[str] = set().union(*(repaired(c, column) for c in chosen))
        flags = sites.assign(hit=sites["site_id"].isin(found))
        per_env = flags.groupby("environment")["hit"].mean()
        return {
            "cells": [cell_key(*c) for c in chosen],
            "mean": float(per_env.mean()),
            "pooled": float(flags["hit"].mean()),
            "repaired_sites": len(found),
            "per_environment": {str(k): float(v) for k, v in per_env.items()},
            "by_error_type": {
                str(k): float(v) for k, v in flags.groupby("site_kind")["hit"].mean().items()
            },
        }

    gen_rows = candidates[
        candidates.set_index(["method", "condition"]).index.isin(list(PRIMARY_CELLS))
    ]
    distinct = gen_rows.drop_duplicates(["site_id", "candidate_text"])
    best_single = max(
        (effect_row for effect_row in (_rates(outcomes, c)["mean"] for c in cells)), default=0.0
    )
    union_all = union_rate(cells, "exact_any")
    top1_union = union_rate(cells, "exact_at_1")
    _write_json_once(
        UNION_CEILING,
        {
            **_analysis_envelope("union_generation_ceiling"),
            "is_deployable_method": False,
            "union_all_gen1_cells_any_candidate": union_all,
            "union_all_gen1_cells_top1": top1_union,
            "union_by_model_any_candidate": {
                model: union_rate([(model, c) for c in CELL_CONDITIONS[model]], "exact_any")
                for model in GEN1_MODELS
            },
            "union_with_m0_any_candidate": union_rate([*cells, (M0, XC1_ORACLE)], "exact_any"),
            "best_single_cell_exact_at_1": best_single,
            "union_top1_minus_best_single_top1": top1_union["mean"] - best_single,
            "union_any_candidate_minus_best_single_top1": union_all["mean"] - best_single,
            "burden": {
                "distinct_candidates": len(distinct),
                "distinct_harmful_candidates": int(distinct["is_harmful"].sum()),
                "distinct_candidates_per_site": _ratio(len(distinct), len(sites)),
                "candidates_listed_per_site": _ratio(len(gen_rows), len(sites)),
            },
            "note": (
                "section 58: complementarity is recorded, never promoted to a method. The union "
                "consumes every cell's generation and every listed candidate."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    union_mean = union_all["mean"]
    print(f"overlap: union of GEN1 cells {union_mean:.4f} against best single {best_single:.4f}")
    return 0


# ------------------------------------------------------------------ section 39: cost


def run_cost() -> int:
    """Section 39: measured resource use per shard, model and cell. No financial cost is made up."""
    started = time.monotonic()
    _forbid(RUNTIME_COST)
    rows = []
    for model, condition in _model_cells():
        for spec in environment_specs():
            sidecar = cc_read_json(_sidecar(_shard_path(model, condition, spec["environment"])))
            rows.append(sidecar)
    frame = pd.DataFrame(rows)
    cells = []
    for (model, condition), block in frame.groupby(["model", "condition"], sort=True):
        seconds = float(block["wall_clock_seconds"].sum())
        cells.append(
            {
                "method": model,
                "condition": condition,
                "shards": len(block),
                "requests": int(block["requests"].sum()),
                "wall_clock_seconds": seconds,
                "wall_clock_hours": seconds / 3600.0,
                "requests_per_second": _ratio(int(block["requests"].sum()), seconds),
                "prompt_tokens": int(block["prompt_tokens"].sum()),
                "output_tokens": int(block["output_tokens"].sum()),
                "unfinished_decodes": int(block["unfinished_decodes"].sum()),
                "batch_size": int(block["batch_size"].iloc[0]),
                "peak_device_memory_bytes": (
                    int(block["peak_device_memory_bytes"].max())
                    if block["peak_device_memory_bytes"].notna().any()
                    else None
                ),
                "device": str(block["device"].iloc[0]),
                "allocator_environments": sorted(
                    {
                        json.dumps(value, sort_keys=True)
                        for value in block.get("allocator_environment", pd.Series(dtype=object))
                        if isinstance(value, dict)
                    }
                ),
            }
        )
    probes = {m: cc_read_json(PROBE_DIR / f"{m}.json") for m in GEN1_MODELS}
    audits = {m: cc_read_json(AUDIT_DIR / f"{m}.json") for m in GEN1_MODELS}
    seconds = {
        model: float(frame.loc[frame["model"] == model, "wall_clock_seconds"].sum())
        for model in GEN1_MODELS
    }
    total_seconds = float(frame["wall_clock_seconds"].sum())
    models = {
        model: {
            "parameters_millions": probes[model]["parameters_millions"],
            "weight_bytes": probes[model]["weight_bytes"],
            "generation_seconds": seconds[model],
            "generation_hours": seconds[model] / 3600.0,
            "requests": int(frame.loc[frame["model"] == model, "requests"].sum()),
            "audit_seconds": float(audits[model]["audit_seconds"]),
        }
        for model in GEN1_MODELS
    }
    _write_json_once(
        RUNTIME_COST,
        {
            **_envelope("runtime_cost"),
            "cells": cells,
            "models": models,
            "total_generation_seconds": total_seconds,
            "total_generation_hours": total_seconds / 3600.0,
            "total_requests": int(frame["requests"].sum()),
            "external_api_calls": 0,
            "execution_incidents": list(EXECUTION_INCIDENTS),
            "financial_cost": None,
            "financial_cost_note": "local inference only; no bill exists and none is estimated",
            "clock_note": (
                "time.monotonic, which does not advance while the machine sleeps, so a suspended "
                "session is not counted as compute"
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    hours = float(frame["wall_clock_seconds"].sum()) / 3600
    print(f"cost: {int(frame['requests'].sum())} requests, {hours:.2f} h")
    return 0


# ------------------------------------------------------------------ sections 43-48: controls


def _marker_decoy_proposals() -> dict[str, Any]:
    """How often a model rewrites a clean position when the marker says it is wrong. GT-blind
    past the decoy choice: it compares each answer with the decoy's own OCR text."""
    controls = pd.read_parquet(CONTROL_CONTEXTS)
    decoys = controls[controls["condition"] == X_MARK].set_index("site_id")["decoy_text"]
    answered = changed = 0
    for spec in environment_specs():
        name = spec["environment"]
        raw = pd.read_parquet(_shard_path(M3, X_MARK, name))
        requests = _requests_for(M3, X_MARK, name).set_index("request_id")
        for row in raw.itertuples(index=False):
            parsed = parse_candidates(str(row.output), bool(row.finished))
            if parsed["status"] not in ("strict", "lenient"):
                continue
            answered += 1
            decoy = str(decoys[str(requests.loc[str(row.request_id), "site_id"])]).strip()
            changed += int(any(c.strip() != decoy for c in parsed["candidates"]))
    return {
        "valid_answers": answered,
        "answers_proposing_a_change_at_the_decoy": changed,
        "false_marker_proposal_rate": _ratio(changed, answered),
    }


def run_controls() -> int:
    """Sections 43-48. Each control's expectation is fixed here and tested in --negative."""
    started = time.monotonic()
    _forbid(CONTROL_RESULTS)
    outcomes = _outcomes()
    rates = {
        cell_key(*c): _rates(outcomes, c)
        for c in [*PLAIN_CONTROL_CELLS, *MODEL_CONTROL_CELLS, *PRIMARY_CELLS, (M0, XC1_ORACLE)]
    }
    random5 = _rates(outcomes, (C_RAND, KNOWN_SITE), "exact_at_5")
    subsample: dict[str, Any] = {}
    for model, condition, reference in CONTROL_CELLS:
        subsample[cell_key(model, condition)] = {
            **effect(outcomes, (model, condition), (model, reference)),
            "reference": cell_key(model, reference),
            "reference_on_subsample": effect(outcomes, (model, condition), (model, reference))[
                "b_mean"
            ],
        }
    duplicated: dict[str, Any] = {}
    candidates = pd.read_parquet(GEN_CANDIDATES)
    for model, condition in PRIMARY_CELLS:
        rows = candidates[(candidates["method"] == model) & (candidates["condition"] == condition)]
        doubled = pd.concat([rows, rows], ignore_index=True)
        duplicated[cell_key(model, condition)] = {
            str(k): [
                int(rows.loc[rows["exact"] & (rows["generator_rank"] < k), "site_id"].nunique()),
                int(
                    doubled.loc[
                        doubled["exact"] & (doubled["generator_rank"] < k), "site_id"
                    ].nunique()
                ),
            ]
            for k in K_GRID
        }
    monotone = all(
        bool(
            (outcomes["exact_at_1"] <= outcomes["exact_at_3"]).all()
            and (outcomes["exact_at_3"] <= outcomes["exact_at_5"]).all()
        )
        for _ in (0,)
    )
    _write_json_once(
        CONTROL_RESULTS,
        {
            **_analysis_envelope("control_results"),
            "gt_corrector": {
                **rates[cell_key(C_GT, KNOWN_SITE)],
                "analysis_only": True,
                "uses_gt_replacement": True,
                "control_only": True,
                "expectation": (
                    "exact at every sampled site: the representation can hold the answer"
                ),
            },
            "identity": {
                **rates[cell_key(C_ID, KNOWN_SITE)],
                "expectation": "no exact repair at a known error site",
            },
            "random_replacement": {
                **rates[cell_key(C_RAND, KNOWN_SITE)],
                "exact_at_5": random5,
                "expectation": "top-5 exact generation below M0's known-site top-1",
                "m0_known_site_top1": rates[cell_key(M0, XC1_ORACLE)]["mean"],
            },
            "model_controls": subsample,
            "false_marker": _marker_decoy_proposals(),
            "duplication": {
                "rule": "every candidate of a cell duplicated at the analysis layer",
                "repaired_sites_at_k_before_and_after": duplicated,
                "unchanged": all(
                    before == after
                    for record in duplicated.values()
                    for before, after in record.values()
                ),
            },
            "truncation": {
                "rule": "exact@1 <= exact@3 <= exact@5 at every site of every cell",
                "monotone": monotone,
            },
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        "controls: GT "
        + f"{rates[cell_key(C_GT, KNOWN_SITE)]['mean']:.4f}, identity "
        + f"{rates[cell_key(C_ID, KNOWN_SITE)]['mean']:.4f}, random@5 {random5['mean']:.4f}"
    )
    return 0


# ------------------------------------------------------------------ section 68: falsification


INPUT_BUILDERS = (
    "mark_text",
    "e0_text",
    "line_text",
    "line_window",
    "user_message",
    "crop_box",
    "build_contexts",
    "_cell_frame",
    "_requests_for",
    "run_requests",
    "run_generate",
    "_page_index",
)


def _source_names(function: Callable[..., Any]) -> set[str]:
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


def _prompts_rebuild() -> dict[str, Any]:
    """Every shard's prompts, rebuilt from the frozen GT-blind tables alone, must hash the same."""
    checked = differing = 0
    for model, condition in _model_cells():
        for spec in environment_specs():
            name = spec["environment"]
            raw = pd.read_parquet(_shard_path(model, condition, name))
            requests = _requests_for(model, condition, name).set_index("request_id")
            for row in raw.itertuples(index=False):
                request = requests.loc[str(row.request_id)]
                digest = canonical_hash(
                    {
                        "system": SYSTEM_PROMPT,
                        "user": str(request["user_prompt"]),
                        "image": request["image_sha256"],
                    }
                )
                checked += 1
                differing += int(digest != str(row.prompt_sha256))
    return {"prompts_checked": checked, "prompts_differing": differing}


def run_negative() -> int:
    """Section 68: thirteen tests that would fail if a claim this stage makes were false."""
    started = time.monotonic()
    _forbid(NEGATIVE_TESTS)
    outcomes = _outcomes()
    controls = cc_read_json(CONTROL_RESULTS)
    gains = cc_read_json(VISUAL_GAIN)
    freeze = cc_read_json(RESEARCH_FREEZE)
    tests: list[dict[str, Any]] = []

    def add(name: str, passed: bool, evidence: Any, applicable: bool = True) -> None:
        tests.append(
            {"test": name, "passed": bool(passed), "applicable": applicable, "evidence": evidence}
        )

    gt = controls["gt_corrector"]
    c3 = freeze["sgv_xc1"]["c3_ground_truth_corrector"]
    add(
        "f1_gt_corrector_reaches_the_representation_ceiling",
        gt["repaired_sites"] == gt["sites"]
        and c3["repaired_error_sites"] == c3["expressible_labelable_error_sites"],
        {"sample": [gt["repaired_sites"], gt["sites"]], "population": c3},
    )
    identity = controls["identity"]
    add("f2_identity_repairs_nothing", identity["repaired_sites"] == 0, identity["repaired_sites"])
    random_rate = controls["random_replacement"]["exact_at_5"]["mean"]
    m0 = controls["random_replacement"]["m0_known_site_top1"]
    add(
        "f3_random_replacements_do_not_manufacture_exact_repairs",
        random_rate < MEANINGFUL_DELTA and random_rate < m0,
        {"random_exact_at_5": random_rate, "m0_known_site_top1": m0},
    )
    context = controls["model_controls"][cell_key(M3, X_CTX)]
    add(
        "f4_context_shuffle_reduces_context_dependent_performance",
        context["mean_delta"] < 0,
        {"shuffled_minus_real": context["mean_delta"], "real_on_subsample": context["b_mean"]},
        applicable=context["b_mean"] > 0,
    )
    image = controls["model_controls"][cell_key(M5, X_IMG)]
    visual = next(iter(gains["effects"].values()))["mean_delta"]
    add(
        "f5_image_shuffle_reduces_image_dependent_performance",
        image["mean_delta"] < 0 if visual > 0 else True,
        {"shuffled_minus_real": image["mean_delta"], "visual_gain": visual},
        applicable=visual > 0,
    )
    marker = controls["model_controls"][cell_key(M3, X_MARK)]
    add(
        "f6_moving_the_marker_reduces_targeted_correction",
        marker["mean_delta"] < 0,
        {"moved_minus_real": marker["mean_delta"], "real_on_subsample": marker["b_mean"]},
        applicable=marker["b_mean"] > 0,
    )
    add(
        "f7_adding_candidates_cannot_reduce_top_k",
        bool(controls["truncation"]["monotone"]),
        controls["truncation"],
    )
    add(
        "f8_duplicate_candidates_do_not_raise_top_k",
        bool(controls["duplication"]["unchanged"]),
        controls["duplication"]["unchanged"],
    )
    exceed = []
    for cell in [*PRIMARY_CELLS, *MODEL_CONTROL_CELLS, *BASELINE_CELLS, *PLAIN_CONTROL_CELLS]:
        block = _block(outcomes, cell)
        if int(block["exact_at_5"].sum()) > len(block) or block["site_id"].duplicated().any():
            exceed.append(cell_key(*cell))
    add("f9_exact_generation_cannot_exceed_eligible_sites", not exceed, exceed)
    tables = {
        _relative(path): [c for c in FORBIDDEN_INPUT_COLUMNS if c in pd.read_parquet(path).columns]
        for path in (SAMPLE_SITES, CONTEXTS, CONTROL_CONTEXTS)
    }
    sources = {
        name: sorted(_source_names(globals()[name]) & set(FORBIDDEN_INPUT_COLUMNS))
        for name in INPUT_BUILDERS
    }
    rebuild = _prompts_rebuild()
    add(
        "f10_ground_truth_never_reaches_a_model_input",
        not any(tables.values())
        and not any(sources.values())
        and rebuild["prompts_differing"] == 0,
        {"input_tables": tables, "builder_sources": sources, "prompt_rebuild": rebuild},
    )
    labelled_at = cc_read_json(NORMALIZED_CANDIDATES)["issued_utc"]
    sidecars = [
        cc_read_json(_sidecar(_shard_path(m, c, s["environment"])))
        for m, c in _model_cells()
        for s in environment_specs()
    ]
    first_shard = min(record["issued_utc"] for record in sidecars)
    prompt_at = cc_read_json(PROMPT_REGISTRY)["issued_utc"]
    add(
        "f11_prompt_registry_frozen_before_generation_and_labels",
        prompt_at <= first_shard and prompt_at <= labelled_at,
        {"prompt_registry": prompt_at, "first_shard": first_shard, "labels": labelled_at},
    )
    probes = {m: cc_read_json(PROBE_DIR / f"{m}.json")["issued_utc"] for m in GEN1_MODELS}
    first_by_model = {
        m: min(r["issued_utc"] for r in sidecars if r["model"] == m) for m in GEN1_MODELS
    }
    registry_at = cc_read_json(MODEL_REGISTRY)["issued_utc"]
    availability_at = cc_read_json(MODEL_AVAILABILITY)["issued_utc"]
    add(
        "f12_model_registry_frozen_before_labels",
        registry_at <= first_shard
        and availability_at <= labelled_at
        and all(probes[m] <= first_by_model[m] for m in GEN1_MODELS),
        {
            "model_registry": registry_at,
            "availability": availability_at,
            "probes": probes,
            "first_shard_by_model": first_by_model,
            "labels": labelled_at,
        },
    )
    envelopes = {}
    for path in (
        ERROR_SITE_INVENTORY,
        NORMALIZED_CANDIDATES,
        FORMAT_COMPLIANCE,
        RAW_OUTPUT_MANIFEST,
        BASELINE_REPRODUCTION,
        EXACT_RESULTS,
        TOPK_RESULTS,
        ERROR_TYPE_ANALYSIS,
        OMISSION_ANALYSIS,
        SEGMENTATION_ANALYSIS,
        SUBSTITUTION_ANALYSIS,
        ENGINE_ANALYSIS,
        DOMAIN_ANALYSIS,
        LANGUAGE_ANALYSIS,
        CAPACITY_GAIN,
        CONTEXT_GAIN,
        VISUAL_GAIN,
        NEAR_MISS,
        EDIT_DAMAGE,
        SPAN_ESCAPE,
        MODEL_OVERLAP,
        UNIQUE_REPAIRS,
        UNION_CEILING,
        CONTROL_RESULTS,
    ):
        record = cc_read_json(path)
        envelopes[path.name] = bool(
            record.get("analysis_only")
            and record.get("oracle_localization")
            and record.get("deployable") is False
        )
    add(
        "f13_oracle_localized_artifacts_cannot_select_a_deployable_method",
        all(envelopes.values()) and bool(envelopes),
        {"analysis_artifacts": len(envelopes), "flagged_correctly": sum(envelopes.values())},
    )
    passed = sum(t["passed"] for t in tests)
    _write_json_once(
        NEGATIVE_TESTS,
        {
            **_analysis_envelope("negative_tests"),
            "tests": tests,
            "passed": passed,
            "total": len(tests),
            "note": (
                "f4-f6 are mechanism probes as well as pipeline checks: a control that fails to "
                "reduce performance says the evidence it removes was not load-bearing. Each "
                "records whether its premise applied."
            ),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"negative: {passed} of {len(tests)} pass")
    return 0


# ------------------------------------------------------------------ sections 50-51: statistics


def _cluster_draws(
    frames: dict[str, pd.DataFrame], pooled: bool = False
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Document-clustered paired bootstrap inside each environment, one draw serving every frame.

    Every frame holds the same sites with columns a and b. Documents are resampled with
    replacement within each environment and the same picks serve every frame, so a statistic that
    combines frames -- the maximum over arms -- is resampled coherently.
    """
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    first = next(iter(frames.values()))
    numerators: dict[str, list[np.ndarray]] = {k: [] for k in frames}
    denominators: list[np.ndarray] = []
    rates: dict[str, list[float]] = {k: [] for k in frames}
    totals: dict[str, list[float]] = {k: [0.0, 0.0] for k in frames}
    for environment in sorted(first["environment"].unique()):
        documents = sorted(first.loc[first["environment"] == environment, "document_id"].unique())
        picks = generator.integers(0, len(documents), size=(BOOTSTRAP_RESAMPLES, len(documents)))
        count = (
            first[first["environment"] == environment]
            .groupby("document_id")
            .size()
            .reindex(documents)
            .to_numpy(float)
        )
        denominators.append(count[picks].sum(axis=1))
        for key, frame in frames.items():
            block = frame[frame["environment"] == environment]
            difference = (
                (block["a"] - block["b"]).groupby(block["document_id"]).sum().reindex(documents)
            ).to_numpy(float)
            numerators[key].append(difference[picks].sum(axis=1))
            rates[key].append(float(difference.sum() / count.sum()))
            totals[key][0] += float(difference.sum())
            totals[key][1] += float(count.sum())
    denominator = np.vstack(denominators)
    draws: dict[str, np.ndarray] = {}
    effects: dict[str, float] = {}
    for key in frames:
        numerator = np.vstack(numerators[key])
        with np.errstate(invalid="ignore", divide="ignore"):
            draws[key] = (
                numerator.sum(axis=0) / denominator.sum(axis=0)
                if pooled
                else (numerator / denominator).mean(axis=0)
            )
        effects[key] = totals[key][0] / totals[key][1] if pooled else float(np.mean(rates[key]))
    return effects, draws


def _test(
    outcomes: pd.DataFrame,
    name: str,
    family: str,
    a: tuple[str, str],
    b: tuple[str, str],
    column: str = "exact_at_1",
    kind: str | None = None,
) -> dict[str, Any]:
    mask = (lambda f: f["site_kind"] == kind) if kind else None
    frame = _paired(outcomes, a, b, column, mask)
    pooled = kind is not None
    effects, draws = _cluster_draws({name: frame}, pooled=pooled)
    per_env = (frame["a"] - frame["b"]).groupby(frame["environment"]).mean()
    return {
        "comparison": name,
        "family": family,
        "a": cell_key(*a),
        "b": cell_key(*b),
        "metric": column,
        "error_type": kind,
        "estimator": "pooled site rate" if pooled else "mean over environments of a site rate",
        **xc1._interval_bounded(draws[name], effects[name]),
        "environments_compared": int(per_env.size),
        "environments_improved": int((per_env > 0).sum()),
        "environments_worsened": int((per_env < 0).sum()),
    }


def run_stats() -> int:
    """Sections 50-51: the pre-registered primary family, Holm-corrected, and its secondaries."""
    started = time.monotonic()
    _forbid(STATISTICAL_TESTS)
    outcomes = _outcomes()
    tests: list[dict[str, Any]] = []
    for name, a, b in PRIMARY_COMPARISONS:
        tests.append(_test(outcomes, name, "primary", a, b))
    frames = {cell_key(*cell): _paired(outcomes, cell, (M0, XC1_ORACLE)) for cell in PRIMARY_CELLS}
    effects, draws = _cluster_draws(frames)
    stacked = np.vstack([draws[k] for k in frames])
    best = max(effects, key=lambda k: effects[k])
    tests.append(
        {
            "comparison": P4_NAME,
            "family": "primary",
            "a": best,
            "b": cell_key(M0, XC1_ORACLE),
            "metric": "exact_at_1",
            "estimator": (
                "max over the GEN1 primary cells of the mean over environments of a site-rate "
                "difference, re-selected in every resample"
            ),
            **xc1._interval_bounded(np.nanmax(stacked, axis=0), effects[best]),
            "selected_cell": best,
            "candidate_cells": list(frames),
        }
    )
    for name, a, b in CAPACITY_EFFECTS:
        if (a, b) != ((M3, E1), (M2G, E1)):
            tests.append(_test(outcomes, name, "secondary_capacity", a, b))
    for name, a, b in EVIDENCE_EFFECTS:
        if (a, b) not in (((M3, E2), (M3, E1)), ((M5, E3), (M5, E1))):
            tests.append(_test(outcomes, name, "secondary_evidence", a, b))
    for name, a, b in JOINT_EFFECTS:
        tests.append(_test(outcomes, name, "secondary_joint", a, b))
    for name, a, b in PRIMARY_COMPARISONS:
        for kind in (*WEAK_ERROR_TYPES, STRONG_ERROR_TYPE):
            tests.append(
                _test(outcomes, f"{name} [{kind}]", "secondary_error_type", a, b, kind=kind)
            )
        for k in K_GRID[1:]:
            tests.append(
                _test(outcomes, f"{name} (K={k})", "secondary_top_k", a, b, f"exact_at_{k}")
            )
    for cell in PRIMARY_CELLS:
        tests.append(
            _test(outcomes, f"{cell_key(*cell)} - {M0}", "secondary_vs_m0", cell, (M0, XC1_ORACLE))
        )
        tests.append(
            _test(
                outcomes, f"{cell_key(*cell)} - {M2}", "secondary_vs_xc1_m2", cell, (M2, XC1_ORACLE)
            )
        )
    for model, condition, reference in CONTROL_CELLS:
        tests.append(
            _test(
                outcomes,
                f"{cell_key(model, condition)} - {cell_key(model, reference)}",
                "secondary_controls",
                (model, condition),
                (model, reference),
            )
        )
    families: dict[str, list[str]] = {}
    for family in sorted({t["family"] for t in tests}):
        members = [t for t in tests if t["family"] == family and "p_value" in t]
        families[family] = [t["comparison"] for t in members]
        adjusted = holm({t["comparison"]: {"p_value": t["p_value"]} for t in members})
        for test in members:
            test["adjusted_p_value"] = float(adjusted[test["comparison"]]["holm_adjusted_p"])
            test["significant_after_holm"] = bool(adjusted[test["comparison"]]["survives_holm"])
    _write_json_once(
        STATISTICAL_TESTS,
        {
            **_analysis_envelope("statistical_tests"),
            "inference_units": ["environment", "document"],
            "candidate_rows_are_not_inferential_units": True,
            "bootstrap": {
                "kind": "document-clustered, paired, resampled within environment",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "interval": "percentile",
                "alpha": ALPHA,
                "p_value": "ocr_risk.stats.bootstrap._bootstrap_p_value: +1 corrected, at most 1",
            },
            "multiplicity": {
                "method": "Holm",
                "primary_family": families.get("primary", []),
                "families": families,
            },
            "tests": tests,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    primary = [t for t in tests if t["family"] == "primary"]
    print(
        "stats: "
        + ", ".join(
            f"{t['comparison']} {t['effect']:+.4f} (p_adj {t['adjusted_p_value']:.4f})"
            for t in primary
        )
    )
    return 0


# ------------------------------------------------------------------ section 64: figures

FIGURE_NOTE = "ANALYSIS ONLY -- known error sites located by the alignment; not a deployable method"
MODEL_LABEL = {
    M0: "M0 frozen (XC1)",
    M1: "M1 ByT5 (XC1)",
    M2: "M2 Qwen2.5-1.5B (XC1)",
    M2G: "Qwen2.5-1.5B",
    M3: "Qwen3-4B",
    M4: "Phi-4-mini",
    M5: "Qwen3-VL-4B",
}
MODEL_COLOR = {
    M0: "#2a78d6",
    M1: "#eb6834",
    M2: "#1baf7a",
    M2G: "#7a9e3a",
    M3: "#8e44ad",
    M4: "#c0392b",
    M5: "#16a085",
}
CONDITION_LABEL = {E0: "E0", E1: "E1", E2: "E2", E3: "E3 image", XC1_ORACLE: ""}
REQUIRED_FIGURES = (
    "exact_generation_by_model.png",
    "exact_generation_by_context.png",
    "capacity_gain.png",
    "context_gain.png",
    "visual_gain.png",
    "exact_generation_by_error_type.png",
    "substitution_generation.png",
    "omission_generation.png",
    "segmentation_generation.png",
    "exact_generation_by_environment.png",
    "exact_generation_by_domain.png",
    "exact_generation_by_engine.png",
    "topk_generation.png",
    "abstention_by_model.png",
    "format_failure_by_model.png",
    "near_miss_distance.png",
    "edit_damage_by_model.png",
    "model_unique_repairs.png",
    "union_generation_ceiling.png",
    "opportunity_vs_compute.png",
)


def _cell_label(key: str) -> str:
    model, condition = key.split("::")
    return f"{MODEL_LABEL[model]} {CONDITION_LABEL.get(condition, condition)}".strip()


def _model_of(key: str) -> str:
    return key.split("::")[0]


def _ci(test: dict[str, Any]) -> tuple[float, float]:
    for low, high in (("ci_low", "ci_high"), ("ci_lower", "ci_upper"), ("lower", "upper")):
        if low in test and high in test:
            return float(test[low]), float(test[high])
    raise PhaseError(f"no interval in {sorted(test)}")


def run_figures() -> int:
    """Section 64: the twenty required figures, each drawn from persisted artifacts alone."""
    started = time.monotonic()
    _forbid(FIGURE_MANIFEST)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.edgecolor": xc1.GRID,
            "axes.labelcolor": xc1.INK,
            "xtick.color": xc1.INK_SECONDARY,
            "ytick.color": xc1.INK_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    exact = cc_read_json(EXACT_RESULTS)
    topk = cc_read_json(TOPK_RESULTS)
    types = cc_read_json(ERROR_TYPE_ANALYSIS)
    omission = cc_read_json(OMISSION_ANALYSIS)
    segmentation = cc_read_json(SEGMENTATION_ANALYSIS)
    substitution = cc_read_json(SUBSTITUTION_ANALYSIS)
    engine = cc_read_json(ENGINE_ANALYSIS)
    domain = cc_read_json(DOMAIN_ANALYSIS)
    capacity = cc_read_json(CAPACITY_GAIN)
    context = cc_read_json(CONTEXT_GAIN)
    visual = cc_read_json(VISUAL_GAIN)
    near = cc_read_json(NEAR_MISS)
    damage = cc_read_json(EDIT_DAMAGE)
    formats = cc_read_json(FORMAT_COMPLIANCE)
    unique = cc_read_json(UNIQUE_REPAIRS)
    union = cc_read_json(UNION_CEILING)
    cost = cc_read_json(RUNTIME_COST)
    stats = cc_read_json(STATISTICAL_TESTS)
    cmap = LinearSegmentedColormap.from_list("gen1", list(xc1.SEQUENTIAL_BLUE))
    blue = xc1.SEQUENTIAL_BLUE
    manifest: dict[str, Any] = {}
    cells = exact["cells"]
    gen1 = list(exact["gen1_cells"])
    shown = gen1 + list(exact["baseline_cells"])
    intervals = {
        (t["a"], t["b"]): t
        for t in stats["tests"]
        if t.get("metric") == "exact_at_1" and not t.get("error_type") and "p_value" in t
    }

    def save(fig: Any, name: str, sources: Sequence[Path]) -> None:
        fig.text(0.01, 0.005, FIGURE_NOTE, fontsize=5.5, color=xc1.INK_SECONDARY)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = FIGURE_DIR / name
        fig.savefig(
            path,
            dpi=150,
            facecolor=xc1.SURFACE,
            metadata={"Software": None},
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close(fig)
        # A source is recorded by the signature of its content without clock fields, so a
        # regeneration that writes the same numbers at a later time records the same source.
        manifest[name] = {
            "sha256": file_sha256(path),
            "sources": {p.name: _signature(p) for p in sources},
        }

    def barh(
        ax: Any,
        labels: Sequence[str],
        values: Sequence[float],
        colors: Sequence[str],
        margin: float | None = None,
        errors: Any = None,
    ) -> None:
        positions = np.arange(len(labels))
        ax.barh(positions, values, color=colors, xerr=errors, ecolor=xc1.INK, capsize=2)
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.axvline(0, color=xc1.INK, linewidth=0.6)
        if margin is not None:
            for x in (margin, -margin):
                ax.axvline(x, color=xc1.INK_SECONDARY, linewidth=0.6, linestyle="--")
        ax.grid(axis="x", color=xc1.GRID, linewidth=0.5)

    def heat(ax: Any, matrix: Any, rows: Sequence[str], columns: Sequence[str]) -> None:
        values = np.asarray(matrix, dtype=float)
        top = max(0.05, float(np.nanmax(values))) if np.isfinite(values).any() else 1.0
        ax.imshow(values, cmap=cmap, vmin=0.0, vmax=top, aspect="auto")
        ax.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
        ax.set_yticks(range(len(rows)), rows)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    shade = "white" if values[i, j] > 0.6 * top else xc1.INK
                    ax.text(
                        j,
                        i,
                        f"{values[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=shade,
                    )

    def heatmap(
        name: str,
        title: str,
        rows: dict[tuple[str, str], float],
        columns: Sequence[str],
        labels: Sequence[str] | None,
        source: Path,
    ) -> None:
        matrix = [[rows.get((k, c), np.nan) for c in columns] for k in shown]
        fig, ax = plt.subplots(figsize=(6.6, 0.28 * len(shown) + 1.9))
        heat(ax, matrix, [_cell_label(k) for k in shown], labels or list(columns))
        ax.set_title(title)
        save(fig, name, [source])

    e1 = [cell_key(m, E1) for m in GEN1_MODELS] + list(exact["baseline_cells"])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    barh(
        ax,
        [_cell_label(k) for k in e1],
        [cells[k]["exact_generation"] for k in e1],
        [MODEL_COLOR[_model_of(k)] for k in e1],
    )
    ax.set_xlabel("ExactGeneration@1, mean over environments")
    ax.set_title(
        "Top-1 exact repair at known error sites: the E1 line, and SGV-XC1's known-site arms"
    )
    save(fig, "exact_generation_by_model.png", [EXACT_RESULTS])

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    barh(
        ax,
        [_cell_label(k) for k in gen1],
        [cells[k]["exact_generation"] for k in gen1],
        [MODEL_COLOR[_model_of(k)] for k in gen1],
    )
    ax.set_xlabel("ExactGeneration@1, mean over environments")
    ax.set_title("Top-1 exact repair by model and evidence condition")
    save(fig, "exact_generation_by_context.png", [EXACT_RESULTS])

    def effect_figure(name: str, effects: dict[str, Any], title: str, source: Path) -> None:
        keys = list(effects)
        values = [float(effects[k]["mean_delta"]) for k in keys]
        lower, upper = [], []
        for key, value in zip(keys, values, strict=True):
            test = intervals.get((effects[key]["a"], effects[key]["b"]))
            low, high = _ci(test) if test else (value, value)
            lower.append(max(0.0, value - low))
            upper.append(max(0.0, high - value))
        fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(keys) + 1.5))
        labels = [
            f"{_cell_label(effects[k]['a'])} minus {_cell_label(effects[k]['b'])}" for k in keys
        ]
        barh(ax, labels, values, [blue[4]] * len(keys), MEANINGFUL_DELTA, [lower, upper])
        ax.set_xlabel(
            "difference in ExactGeneration@1 (95% percentile interval); dashed: the 0.10 margin"
        )
        ax.set_title(title)
        save(fig, name, [source, STATISTICAL_TESTS])

    effect_figure(
        "capacity_gain.png",
        {**capacity["effects"], **capacity["joint_effects"]},
        "Capacity at matched evidence, and the joint contrasts",
        CAPACITY_GAIN,
    )
    effect_figure(
        "context_gain.png",
        context["effects"],
        "Wider textual context inside each model",
        CONTEXT_GAIN,
    )
    visual_effect = next(iter(visual["effects"].values()))
    environments = list(visual_effect["per_environment_delta"])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    barh(
        ax,
        environments,
        [visual_effect["per_environment_delta"][e] for e in environments],
        [MODEL_COLOR[M5]] * len(environments),
        MEANINGFUL_DELTA,
    )
    ax.axvline(visual_effect["mean_delta"], color=xc1.INK, linestyle=":", linewidth=1)
    ax.set_xlabel("E3 minus E1, per environment; dotted: the mean")
    ax.set_title("Visual effect inside one model: Qwen3-VL-4B with the crop minus without it")
    save(fig, "visual_gain.png", [VISUAL_GAIN])

    kinds = ["substitution", "deletion", "segmentation", "insertion"]
    heatmap(
        "exact_generation_by_error_type.png",
        "ExactGeneration@1 by frozen error type, pooled",
        {(r["cell"], r["site_kind"]): r["exact_at_1"] for r in types["rows"]},
        kinds,
        ["substitution", "omission", "segmentation", "spurious insertion"],
        ERROR_TYPE_ANALYSIS,
    )
    heatmap(
        "substitution_generation.png",
        "Substitution sites: ExactGeneration@1 by subtype",
        {(r["cell"], r["subtype"]): r["exact_at_1"] for r in substitution["rows"]},
        [s for s in substitution["subtypes"] if s in substitution["sample_counts"]],
        None,
        SUBSTITUTION_ANALYSIS,
    )
    heatmap(
        "segmentation_generation.png",
        "Segmentation sites: ExactGeneration@1 by subtype",
        {(r["cell"], r["subtype"]): r["exact_at_1"] for r in segmentation["rows"]},
        [s for s in segmentation["subtypes"] if s in segmentation["sample_counts"]],
        None,
        SEGMENTATION_ANALYSIS,
    )
    rows = omission["cells"]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x = np.arange(len(rows))
    ax.bar(x - 0.27, [r["exact_at_1"] for r in rows], 0.27, color=blue[5], label="exact@1")
    ax.bar(x, [r["exact_at_5"] for r in rows], 0.27, color=blue[2], label="exact@5")
    ax.bar(
        x + 0.27,
        [r["abstention_rate"] for r in rows],
        0.27,
        color=xc1.DERIVED_GRAYS[1],
        label="abstention",
    )
    ax.set_xticks(x, [_cell_label(r["cell"]) for r in rows], rotation=35, ha="right")
    ax.legend(frameon=False)
    ax.set_title("Omission sites (text missing from the OCR)")
    save(fig, "omission_generation.png", [OMISSION_ANALYSIS])

    names = [spec["environment"] for spec in environment_specs()]
    heatmap(
        "exact_generation_by_environment.png",
        "ExactGeneration@1 by environment",
        {
            (k, e): cells[k]["per_environment_exact_at_1"].get(e, np.nan)
            for k in shown
            for e in names
        },
        names,
        None,
        EXACT_RESULTS,
    )
    heatmap(
        "exact_generation_by_domain.png",
        "ExactGeneration@1 by document domain, pooled",
        {(r["cell"], r["domain"]): r["exact_at_1"] for r in domain["rows"]},
        ["modern_forms", "historical_print"],
        ["modern forms (FUNSD)", "historical print (OCR-D SBB)"],
        DOMAIN_ANALYSIS,
    )
    engines = sorted({r["base_engine"] for r in engine["by_base_engine"]})
    heatmap(
        "exact_generation_by_engine.png",
        "ExactGeneration@1 by OCR engine family, pooled",
        {(r["cell"], r["base_engine"]): r["exact_at_1"] for r in engine["by_base_engine"]},
        engines,
        None,
        ENGINE_ANALYSIS,
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    styles = {E0: ":", E1: "-", E2: "--", E3: "-."}
    for key in gen1:
        model, condition = key.split("::")
        curve = [topk["cells"][key]["exact_at_k"][str(k)] for k in K_GRID]
        ax.plot(
            K_GRID,
            curve,
            marker="o",
            markersize=3,
            color=MODEL_COLOR[model],
            linestyle=styles[condition],
            label=_cell_label(key),
        )
    ax.set_xticks(K_GRID)
    ax.set_xlabel("K (prefix of the model's own ranked list)")
    ax.set_ylabel("ExactGeneration@K")
    ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Ranking headroom: exact repair within the top K")
    save(fig, "topk_generation.png", [TOPK_RESULTS])

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    barh(
        ax,
        [_cell_label(k) for k in gen1],
        [cells[k]["abstention_rate"] for k in gen1],
        [MODEL_COLOR[_model_of(k)] for k in gen1],
    )
    ax.set_xlabel("share of sites with a valid answer that proposes no change")
    ax.set_title("Abstention at known error sites")
    save(fig, "abstention_by_model.png", [EXACT_RESULTS])

    records = {cell_key(r["method"], r["condition"]): r for r in formats["cells"]}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    left = np.zeros(len(gen1))
    for status, color in (
        ("strict", blue[5]),
        ("lenient", blue[2]),
        ("truncated", "#e0a43b"),
        ("invalid", "#c0392b"),
    ):
        share = np.array(
            [records[k][f"status_{status}"] / max(1, records[k]["answers"]) for k in gen1]
        )
        ax.barh(np.arange(len(gen1)), share, left=left, color=color, label=status)
        left += share
    ax.set_yticks(np.arange(len(gen1)), [_cell_label(k) for k in gen1])
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Answer format: strict JSON, recovered JSON, cut off by the cap, unreadable")
    save(fig, "format_failure_by_model.png", [FORMAT_COMPLIANCE])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    labels = [_cell_label(k) for k in shown]
    colors = [MODEL_COLOR[_model_of(k)] for k in shown]
    barh(
        axes[0], labels, [near["cells"][k]["edit_distance_to_gt"]["median"] for k in shown], colors
    )
    axes[0].set_xlabel("median edit distance, wrong top-1 to GT")
    barh(axes[1], labels, [near["cells"][k]["share_within_one_character"] for k in shown], colors)
    axes[1].set_xlabel("share of wrong top-1 within one character")
    fig.suptitle("Near misses among wrong top-1 candidates", fontsize=8.5)
    save(fig, "near_miss_distance.png", [NEAR_MISS])

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    left = np.zeros(len(shown))
    for part, color in (
        ("improves_not_exact", blue[3]),
        ("equivalent_distance", xc1.DERIVED_GRAYS[0]),
        ("worsens", "#c0392b"),
    ):
        share = np.array(
            [
                damage["cells"][k][f"all_{part}"]
                / max(1, damage["cells"][k]["all_wrong_candidates"])
                for k in shown
            ]
        )
        ax.barh(np.arange(len(shown)), share, left=left, color=color, label=part.replace("_", " "))
        left += share
    ax.set_yticks(np.arange(len(shown)), labels)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("What wrong candidates do to the site, against the OCR's own distance")
    save(fig, "edit_damage_by_model.png", [EDIT_DAMAGE])

    models = list(unique["models"])
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    barh(
        ax,
        [MODEL_LABEL[m] for m in models],
        [unique["models"][m]["unique_repairs"] for m in models],
        [MODEL_COLOR[m] for m in models],
    )
    ax.set_xlabel("sites repaired within the top 5 by this model only")
    ax.set_title("Unique repairs by model")
    save(fig, "model_unique_repairs.png", [UNIQUE_REPAIRS])

    bars = {
        "best single GEN1 cell, top 1": union["best_single_cell_exact_at_1"],
        "union of GEN1 cells, top 1": union["union_all_gen1_cells_top1"]["mean"],
        "union of GEN1 cells, any candidate": union["union_all_gen1_cells_any_candidate"]["mean"],
        "union with M0, any candidate": union["union_with_m0_any_candidate"]["mean"],
    }
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    barh(ax, list(bars), list(bars.values()), [blue[i] for i in (5, 4, 3, 2)], MEANINGFUL_DELTA)
    ax.set_xlabel("exact repair, mean over environments")
    ax.set_title("Union ceiling -- an analysis-only headroom, never a method")
    save(fig, "union_generation_ceiling.png", [UNION_CEILING])

    runtime = {cell_key(r["method"], r["condition"]): r["wall_clock_hours"] for r in cost["cells"]}
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for key in gen1:
        ax.scatter(
            runtime[key], cells[key]["exact_generation"], color=MODEL_COLOR[_model_of(key)], s=18
        )
        ax.annotate(
            _cell_label(key),
            (runtime[key], cells[key]["exact_generation"]),
            fontsize=5.5,
            xytext=(3, 2),
            textcoords="offset points",
        )
    ax.set_xlabel("measured generation hours on this machine")
    ax.set_ylabel("ExactGeneration@1")
    ax.grid(color=xc1.GRID, linewidth=0.5)
    ax.set_title("Exact repair against measured compute")
    save(fig, "opportunity_vs_compute.png", [RUNTIME_COST, EXACT_RESULTS])

    missing = [name for name in REQUIRED_FIGURES if name not in manifest]
    _write_json_once(
        FIGURE_MANIFEST,
        {
            **_analysis_envelope("figure_manifest"),
            "figures": manifest,
            "required": list(REQUIRED_FIGURES),
            "missing": missing,
            "note": FIGURE_NOTE,
            "illustrative_values": False,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    if missing:
        raise PhaseError(f"figures missing: {missing}")
    print(f"figures: {len(manifest)} written")
    return 0


# ------------------------------------------------------------------ sections 57, 61, 76-77: decide

NEXT_STAGES = {
    "A": "NEXT: RELIABILITY TRANSFER TO STRONG CORRECTOR",
    "B": "NEXT: CONTEXT-AWARE CANDIDATE GENERATION + RELIABILITY",
    "C": "NEXT: FREEZE THE JOINT GENERATOR, THEN RELIABILITY TRANSFER",
    "D": "NEXT: TASK-FORMULATION / UNDERDETERMINATION INVESTIGATION",
}
DOMINANT_MECHANISM = {
    "A": "model capacity",
    "B": "context / evidence",
    "C": "capacity and evidence together",
    "D": "none: exact replacement stays weak with the error location known",
}


def assign_outcome(capacity: bool, evidence: bool, joint: bool) -> str | None:
    """The frozen outcome rule. None marks the one combination it leaves open."""
    if capacity and not evidence:
        return "A"
    if evidence and not capacity:
        return "B"
    if joint and capacity == evidence:
        return "C"
    if not (capacity or evidence or joint):
        return "D"
    return None


def run_decide() -> int:
    """The machine-readable decision: exactly one outcome, by the rule frozen in the design."""
    started = time.monotonic()
    _forbid(DECISION)
    exact = cc_read_json(EXACT_RESULTS)
    topk = cc_read_json(TOPK_RESULTS)
    capacity = cc_read_json(CAPACITY_GAIN)
    context = cc_read_json(CONTEXT_GAIN)
    visual = cc_read_json(VISUAL_GAIN)
    stats = cc_read_json(STATISTICAL_TESTS)
    negative = cc_read_json(NEGATIVE_TESTS)
    controls = cc_read_json(CONTROL_RESULTS)
    union = cc_read_json(UNION_CEILING)
    reproduction = cc_read_json(BASELINE_REPRODUCTION)
    types = cc_read_json(ERROR_TYPE_ANALYSIS)
    domains = cc_read_json(DOMAIN_ANALYSIS)
    freeze = cc_read_json(RESEARCH_FREEZE)
    outcomes = _outcomes()
    cells = exact["cells"]
    gen1 = list(exact["gen1_cells"])
    capacity_effects = capacity["effects"]
    joint_effects = capacity["joint_effects"]
    evidence_effects = {**context["effects"], **visual["effects"]}
    capacity_material = any(r["material"] for r in capacity_effects.values())
    evidence_material = any(r["material"] for r in evidence_effects.values())
    joint_material = any(r["material"] for r in joint_effects.values())
    outcome = assign_outcome(capacity_material, evidence_material, joint_material)
    if outcome is None:
        raise PhaseError(
            "a capacity and an evidence effect are both material while no joint effect is; the "
            "frozen rule assigns no outcome to that combination"
        )
    best = str(exact["best_gen1_cell"])
    best_cell = (best.split("::")[0], best.split("::")[1])
    text_cells = [k for k in gen1 if not k.endswith(f"::{E3}")]
    best_text = max(text_cells, key=lambda k: cells[k]["exact_generation"])
    image_cell = cell_key(M5, E3)
    reference = cell_key(M2G, E1)
    m0 = cell_key(M0, XC1_ORACLE)
    versus_small = effect(outcomes, best_cell, (M2G, E1))
    type_rows = {(r["cell"], r["site_kind"]): float(r["exact_at_1"]) for r in types["rows"]}
    domain_rows = {(r["cell"], r["domain"]): float(r["exact_at_1"]) for r in domains["rows"]}
    domain_delta = {
        d: domain_rows[best, d] - domain_rows[reference, d]
        for d in ("modern_forms", "historical_print")
    }
    all_effects = {**capacity_effects, **evidence_effects, **joint_effects}
    largest = max(all_effects, key=lambda k: all_effects[k]["mean_delta"])
    primary = [t for t in stats["tests"] if t["family"] == "primary"]

    def by_type(name: str, kind: str) -> float:
        return float(all_effects[name]["delta_by_error_type"].get(kind, float("nan")))

    sources = {
        "capacity": "capacity_m3_e1",
        "local_context": "context_m3_e1_vs_e0",
        "extended_context": "context_m3_e2_vs_e1",
        "image": "visual_m5_e3_vs_e1",
    }
    q4 = {}
    for kind in WEAK_ERROR_TYPES:
        deltas = {source: by_type(name, kind) for source, name in sources.items()}
        leader = max(deltas, key=lambda s: deltas[s])
        q4[kind] = {
            "delta_by_source": deltas,
            "largest_source": leader,
            "reaches_weak_type_margin": deltas[leader] >= WEAK_TYPE_DELTA,
        }
    omission_capacity = max(by_type(n, "deletion") for n in capacity_effects)
    omission_context = max(by_type(n, "deletion") for n in evidence_effects)
    if max(omission_capacity, omission_context) < WEAK_TYPE_DELTA:
        q5 = "neither: omissions stay essentially unrepaired whatever the model or the evidence"
    elif omission_context > omission_capacity:
        q5 = "lack of context: evidence moves omission repair more than capacity does"
    else:
        q5 = "generation capacity: a stronger model moves omission repair more than evidence does"
    questions = {
        "Q1": {
            "answer": "yes" if capacity_material else "no",
            "effects": {k: v["mean_delta"] for k, v in capacity_effects.items()},
        },
        "Q2": {
            "answer": "yes" if any(r["material"] for r in context["effects"].values()) else "no",
            "effects": {k: v["mean_delta"] for k, v in context["effects"].items()},
        },
        "Q3": {
            "answer": "yes" if any(r["material"] for r in visual["effects"].values()) else "no",
            "effects": {k: v["mean_delta"] for k, v in visual["effects"].items()},
        },
        "Q4": q4,
        "Q5": {
            "answer": q5,
            "largest_capacity_effect_on_omission": omission_capacity,
            "largest_evidence_effect_on_omission": omission_context,
        },
        "Q6": {
            "answer": "yes" if outcome != "D" else "no",
            "best_gen1_exact_generation": cells[best]["exact_generation"],
        },
    }
    if outcome == "D":
        reason = (
            f"outcome D: the strongest GEN1 cell, {best}, reaches ExactGeneration@1 "
            f"{cells[best]['exact_generation']:.4f} at known error sites, against "
            f"{cells[reference]['exact_generation']:.4f} for the small model at matched evidence "
            f"and {cells[m0]['exact_generation']:.4f} for M0; no capacity, evidence or joint "
            f"effect is material -- the largest, {largest}, is "
            f"{all_effects[largest]['mean_delta']:+.4f} "
            f"where {MEANINGFUL_DELTA} with breadth and weak-type recovery was required."
        )
    else:
        reason = (
            f"outcome {outcome}: material effects -- capacity {capacity_material}, evidence "
            f"{evidence_material}, joint {joint_material}; the strongest GEN1 cell, {best}, "
            f"reaches ExactGeneration@1 {cells[best]['exact_generation']:.4f} against "
            f"{cells[reference]['exact_generation']:.4f} for the small model at matched evidence."
        )
    _write_json_once(
        DECISION,
        {
            **_analysis_envelope("research_decision"),
            "status": "COMPLETE",
            "stage_kind": "DEVELOPMENT / MECHANISM",
            "primary_metric": "ExactGeneration@1 at oracle-located known error sites",
            "primary_epsilon_if_applicable": None,
            "models_evaluated": [*GEN1_MODELS, *XC1_BASELINES],
            "contexts_evaluated": list(EVIDENCE),
            "frozen_xc1_baseline": {
                "m0_known_site_top1_on_sample": cells[m0]["exact_generation"],
                "m0_candidate_opportunity": freeze["sgv_xc1"]["headline"]["m0_opportunity"],
                "oracle_localization_by_method": freeze["sgv_xc1"]["oracle_localization_by_method"],
                "reproduction_checks_agreeing": reproduction["checks_agreeing"],
                "reproduction_checks_total": reproduction["checks_total"],
            },
            "best_gen1_cell": best,
            "best_text_model": _model_of(best_text),
            "best_text_context": best_text.split("::")[1],
            "best_text_exact_generation": cells[best_text]["exact_generation"],
            "best_multimodal_model": M5,
            "best_multimodal_exact_generation": cells[image_cell]["exact_generation"],
            "capacity_gain": capacity_effects["capacity_m3_e1"]["mean_delta"],
            "context_gain": context["effects"]["context_m3_e2_vs_e1"]["mean_delta"],
            "local_context_gain": context["effects"]["context_m3_e1_vs_e0"]["mean_delta"],
            "visual_gain": visual["effects"]["visual_m5_e3_vs_e1"]["mean_delta"],
            "effects_material": {
                "capacity": capacity_material,
                "evidence": evidence_material,
                "joint": joint_material,
            },
            "largest_effect": {"name": largest, "mean_delta": all_effects[largest]["mean_delta"]},
            "substitution_exact_generation": type_rows.get((best, "substitution")),
            "omission_exact_generation": type_rows.get((best, "deletion")),
            "segmentation_exact_generation": type_rows.get((best, "segmentation")),
            "spurious_insertion_exact_generation": type_rows.get((best, "insertion")),
            "environments_improved": versus_small["environments_improved"],
            "environments_meaningfully_improved": versus_small[
                "environments_meaningfully_improved"
            ],
            "domains_improved": [d for d, v in domain_delta.items() if v > 0],
            "domains_meaningfully_improved": [
                d for d, v in domain_delta.items() if v >= MEANINGFUL_DELTA
            ],
            "domain_delta_vs_small_model": domain_delta,
            "top1_gain": versus_small["mean_delta"],
            "top5_gain": topk["cells"][best]["exact_at_k"]["5"]
            - topk["cells"][reference]["exact_at_k"]["5"],
            "top1_gain_vs_m0": cells[best]["exact_generation"] - cells[m0]["exact_generation"],
            "candidate_burden": {
                "candidates_per_site": cells[best]["candidates_per_site"],
                "harmful_share_of_candidates": cells[best]["harmful_share_of_candidates"],
                "candidate_precision": cells[best]["candidate_precision"],
            },
            "abstention_rate": cells[best]["abstention_rate"],
            "union_any_candidate": union["union_all_gen1_cells_any_candidate"]["mean"],
            "dominant_mechanism": DOMINANT_MECHANISM[outcome],
            "outcome": outcome,
            "outcome_label": OUTCOME_TAXONOMY[outcome],
            "outcome_rule": OUTCOME_RULE,
            "questions": questions,
            "statistical_support": [
                {
                    k: t.get(k)
                    for k in ("comparison", "a", "b", "effect", "p_value", "adjusted_p_value")
                }
                | {"interval": list(_ci(t))}
                for t in primary
            ],
            "controls": {
                "gt_corrector_exact": controls["gt_corrector"]["mean"],
                "identity_exact": controls["identity"]["mean"],
                "random_exact_at_5": controls["random_replacement"]["exact_at_5"]["mean"],
                "false_marker_proposal_rate": controls["false_marker"][
                    "false_marker_proposal_rate"
                ],
            },
            "falsification_tests_passed": negative["passed"],
            "falsification_tests_total": negative["total"],
            "ready_for_reliability_transfer": outcome in ("A", "B", "C"),
            "ready_for_external_confirmation": False,
            "confirmatory_reserve_consumed": False,
            "deployable": False,
            "recommended_next_stage": NEXT_STAGES[outcome],
            "reason": reason,
            "issued_head": _git("rev-parse", "HEAD"),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(f"decide: outcome {outcome} -- {NEXT_STAGES[outcome]}")
    return 0


# ------------------------------------------------------------------ section 73: determinism

DERIVED_OUTPUTS = (
    "GEN_CANDIDATES",
    "SITE_OUTCOMES",
    "XC1_RELABEL",
    "RAW_OUTPUT_MANIFEST",
    "RAW_OUTPUT_HASHES",
    "NORMALIZED_CANDIDATES",
    "FORMAT_COMPLIANCE",
    "BASELINE_REPRODUCTION",
    "EXACT_RESULTS",
    "TOPK_RESULTS",
    "ERROR_TYPE_ANALYSIS",
    "OMISSION_ANALYSIS",
    "SEGMENTATION_ANALYSIS",
    "SUBSTITUTION_ANALYSIS",
    "ENGINE_ANALYSIS",
    "DOMAIN_ANALYSIS",
    "LANGUAGE_ANALYSIS",
    "CONTEXT_GAIN",
    "CAPACITY_GAIN",
    "VISUAL_GAIN",
    "NEAR_MISS",
    "EDIT_DAMAGE",
    "SPAN_ESCAPE",
    "MODEL_OVERLAP",
    "UNIQUE_REPAIRS",
    "UNION_CEILING",
    "RUNTIME_COST",
    "CONTROL_RESULTS",
    "NEGATIVE_TESTS",
    "STATISTICAL_TESTS",
    "FIGURE_DIR",
    "FIGURE_MANIFEST",
    "DECISION",
)
TABLE_KEYS = {
    "generation_candidates.parquet": ["method", "condition", "candidate_id"],
    "site_outcomes.parquet": ["method", "condition", "site_id"],
    "xc1_m2_relabel.parquet": ["candidate_id"],
}
VOLATILE_KEYS = frozenset(
    {"issued_utc", "runtime_seconds", "audit_seconds", "probed_seconds", "head", "issued_head"}
)
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
DETERMINISM_ENVIRONMENTS = ("funsd/doctr", "sbb/doctr")


def _stable_view(value: Any) -> Any:
    """An artifact with its clock and host fields removed, so two regenerations compare exactly."""
    if isinstance(value, dict):
        return {k: _stable_view(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_stable_view(v) for v in value]
    if isinstance(value, str) and _UTC.match(value):
        return "<utc>"
    if isinstance(value, float) and value != value:
        return "nan"
    return value


def _signature(path: Path) -> str:
    if path.suffix == ".json":
        return canonical_hash(_stable_view(cc_read_json(path)))
    frame = pd.read_parquet(path)
    keys = TABLE_KEYS.get(path.name, list(frame.columns))
    frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
    return canonical_hash(frame.to_json(orient="records", default_handler=str))


def _derived_phases() -> tuple[tuple[str, Callable[[], int]], ...]:
    return (
        ("normalize", run_normalize),
        ("reproduce", run_reproduce),
        ("exact", run_exact),
        ("strata", run_strata),
        ("gains", run_gains),
        ("diagnostics", run_diagnostics),
        ("overlap", run_overlap),
        ("cost", run_cost),
        ("controls", run_controls),
        ("negative", run_negative),
        ("stats", run_stats),
        ("figures", run_figures),
        ("decide", run_decide),
    )


def _current_signatures() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in DERIVED_OUTPUTS:
        path = globals()[name]
        if isinstance(path, Path) and path.is_file():
            out[path.name] = _signature(path)
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


def _context_rebuild() -> dict[str, Any]:
    """GT-blind inputs rebuilt from the raw OCR and page images, against the frozen tables."""
    sample = pd.read_parquet(SAMPLE_SITES)
    frozen = pd.read_parquet(CONTEXTS).set_index("site_id")
    records = []
    for name in DETERMINISM_ENVIRONMENTS:
        sites = sample[sample["environment"] == name]
        rebuilt = build_contexts(
            sites, _page_index(_spec(name)), CACHE / "determinism" / "crops" / _slug(name)
        ).set_index("site_id")
        text = all(
            bool((rebuilt[c] == frozen.loc[rebuilt.index, c]).all())
            for c in ("e0_text", "e1_text", "e2_text")
        )
        crops = bool((rebuilt["crop_sha256"] == frozen.loc[rebuilt.index, "crop_sha256"]).all())
        records.append(
            {
                "environment": name,
                "sites": len(rebuilt),
                "text_identical": text,
                "crops_identical": crops,
            }
        )
    return {
        "environments": records,
        "identical": all(r["text_identical"] and r["crops_identical"] for r in records),
    }


def run_determinism() -> int:
    """Section 73: two independent downstream regenerations, the inputs, and the models."""
    started = time.monotonic()
    _forbid(DETERMINISM)
    original = _current_signatures()
    runs = [_sandboxed(CACHE / "determinism" / f"run_{i}") for i in range(2)]
    differing = sorted(
        {name for run in runs for name in original if run.get(name) != original[name]}
    )
    inputs = _context_rebuild()
    regeneration = {
        m: cc_read_json(REGENERATION_DIR / f"{m}.json")
        for m in GEN1_MODELS
        if (REGENERATION_DIR / f"{m}.json").is_file()
    }
    models_identical = len(regeneration) == len(GEN1_MODELS) and all(
        r["identical"] for r in regeneration.values()
    )
    identical = not differing and inputs["identical"] and models_identical
    _write_json_once(
        DETERMINISM,
        {
            **_envelope("determinism"),
            "compared": [
                "parsed answers, normalized candidates and labels, regenerated from the raw cache",
                "exact, error-type, gain, diagnostic, control and statistical artifacts",
                "figures, by content hash",
                "the decision",
                "the GT-blind contexts and crops, rebuilt from the raw OCR and the page images",
                "each model's first batches, regenerated while its checkpoint was on disk",
            ],
            "derived_regenerations": len(runs),
            "derived_artifacts_compared": len(original),
            "differing_artifacts": differing,
            "derived_runs_identical": not differing,
            "input_rebuild": inputs,
            "model_regeneration": regeneration,
            "model_regeneration_identical": models_identical,
            "all_runs_identical": identical,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    print(
        f"determinism: {len(original)} artifacts x {len(runs)} regenerations, identical {identical}"
    )
    if not identical:
        raise PhaseError(f"determinism failed: {differing}, inputs {inputs['identical']}")
    return 0


# ------------------------------------------------------------------ section 72: record

PRODUCED: tuple[Path, ...] = (
    RESEARCH_FREEZE,
    FROZEN_CONFIGURATION,
    ENVIRONMENT_INVENTORY,
    DESIGN_RECORD,
    ERROR_SITE_INVENTORY,
    MODEL_AVAILABILITY,
    PROMPT_FORMAT_AUDIT,
    MODEL_REGISTRY,
    MODEL_VERSIONS,
    CONTEXT_REGISTRY,
    PROMPT_REGISTRY,
    DECODING_REGISTRY,
    CROP_REGISTRY,
    SAMPLE_SITES,
    SAMPLE_TRUTH,
    CONTEXTS,
    CONTROL_CONTEXTS,
    GEN_CANDIDATES,
    SITE_OUTCOMES,
    RAW_OUTPUT_MANIFEST,
    RAW_OUTPUT_HASHES,
    NORMALIZED_CANDIDATES,
    FORMAT_COMPLIANCE,
    BASELINE_REPRODUCTION,
    EXACT_RESULTS,
    TOPK_RESULTS,
    ERROR_TYPE_ANALYSIS,
    OMISSION_ANALYSIS,
    SEGMENTATION_ANALYSIS,
    SUBSTITUTION_ANALYSIS,
    ENGINE_ANALYSIS,
    DOMAIN_ANALYSIS,
    LANGUAGE_ANALYSIS,
    CONTEXT_GAIN,
    CAPACITY_GAIN,
    VISUAL_GAIN,
    NEAR_MISS,
    EDIT_DAMAGE,
    SPAN_ESCAPE,
    MODEL_OVERLAP,
    UNIQUE_REPAIRS,
    UNION_CEILING,
    RUNTIME_COST,
    CONTROL_RESULTS,
    NEGATIVE_TESTS,
    STATISTICAL_TESTS,
    FIGURE_MANIFEST,
    DETERMINISM,
    DECISION,
    PROVENANCE,
    TRACEABILITY,
)

REPORT_SECTIONS: dict[str, tuple[Path, ...]] = {
    "1. Motivation": (RESEARCH_FREEZE,),
    "2. Frozen SGV-XC1 Result": (RESEARCH_FREEZE,),
    "3. Research Question": (DESIGN_RECORD,),
    "4. Why Localization Is Removed": (DESIGN_RECORD, RESEARCH_FREEZE),
    "5. Non-Goals": (DESIGN_RECORD,),
    "6. Known-Site Experimental Design": (DESIGN_RECORD, ERROR_SITE_INVENTORY),
    "7. Ground-Truth Isolation": (NEGATIVE_TESTS, DESIGN_RECORD, CONTEXT_REGISTRY),
    "8. Model Family": (MODEL_REGISTRY, MODEL_VERSIONS, MODEL_AVAILABILITY),
    "9. Model Availability Audit": (MODEL_AVAILABILITY, PROMPT_FORMAT_AUDIT),
    "10. Evidence Conditions": (CONTEXT_REGISTRY, CROP_REGISTRY, FORMAT_COMPLIANCE),
    "11. Prompt and Decoding Freeze": (
        PROMPT_REGISTRY,
        DECODING_REGISTRY,
        PROMPT_FORMAT_AUDIT,
        SUPERSEDED_AUDIT,
    ),
    "12. Atomic-Edit Representation": (
        FROZEN_CONFIGURATION,
        NORMALIZED_CANDIDATES,
        FORMAT_COMPLIANCE,
    ),
    "13. Baseline Reproduction": (BASELINE_REPRODUCTION,),
    "14. Overall Exact Generation": (EXACT_RESULTS,),
    "15. Capacity Effect": (CAPACITY_GAIN, STATISTICAL_TESTS),
    "16. Local Context Effect": (CONTEXT_GAIN, STATISTICAL_TESTS),
    "17. Extended Context Effect": (CONTEXT_GAIN, STATISTICAL_TESTS),
    "18. Visual Evidence Effect": (VISUAL_GAIN, STATISTICAL_TESTS),
    "19. Substitution": (SUBSTITUTION_ANALYSIS, ERROR_TYPE_ANALYSIS),
    "20. Omission": (OMISSION_ANALYSIS, ERROR_TYPE_ANALYSIS),
    "21. Segmentation": (SEGMENTATION_ANALYSIS, ERROR_TYPE_ANALYSIS),
    "22. Spurious Insertion": (ERROR_TYPE_ANALYSIS,),
    "23. Engine Analysis": (ENGINE_ANALYSIS,),
    "24. Domain Analysis": (DOMAIN_ANALYSIS,),
    "25. Language / Typography Analysis": (LANGUAGE_ANALYSIS,),
    "26. Top-K Headroom": (TOPK_RESULTS,),
    "27. Abstention": (EXACT_RESULTS, FORMAT_COMPLIANCE),
    "28. Wrong-Edit Damage": (EDIT_DAMAGE,),
    "29. Near Misses": (NEAR_MISS,),
    "30. Model Complementarity": (MODEL_OVERLAP, UNIQUE_REPAIRS),
    "31. Union Ceiling": (UNION_CEILING,),
    "32. Controls": (CONTROL_RESULTS,),
    "33. Falsification Tests": (NEGATIVE_TESTS, DETERMINISM),
    "34. Statistics": (STATISTICAL_TESTS,),
    "35. Compute / Runtime": (RUNTIME_COST,),
    "36. Limitations": (
        DESIGN_RECORD,
        ERROR_SITE_INVENTORY,
        MODEL_AVAILABILITY,
        FORMAT_COMPLIANCE,
        VISUAL_GAIN,
        STATISTICAL_TESTS,
        SUBSTITUTION_ANALYSIS,
        CONTROL_RESULTS,
        RUNTIME_COST,
    ),
    "37. Research Decision": (DECISION, VISUAL_GAIN, STATISTICAL_TESTS),
    "38. Implications for Candidate Generation": (
        DECISION,
        UNION_CEILING,
        RESEARCH_FREEZE,
        EXACT_RESULTS,
        CONTROL_RESULTS,
        TOPK_RESULTS,
    ),
    "39. Next Stage": (DECISION,),
}


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
    rederived: list[bool] = []
    for run in range(2):
        sandbox = CACHE / "record" / f"decision_{run}.json"
        sandbox.parent.mkdir(parents=True, exist_ok=True)
        sandbox.unlink(missing_ok=True)
        saved = globals()["DECISION"]
        try:
            globals()["DECISION"] = sandbox
            run_decide()
        finally:
            globals()["DECISION"] = saved
        rederived.append(
            _stable_view(cc_read_json(sandbox)) == _stable_view(cc_read_json(DECISION))
        )
        sandbox.unlink(missing_ok=True)
    if not all(rederived):
        raise PhaseError("the research decision does not re-derive identically")
    freeze = cc_read_json(RESEARCH_FREEZE)
    changed = {
        group: sorted(p for p, d in freeze[group].items() if file_sha256(REPO / p) != d)
        for group in ("sgv_xc1_artifacts", "sgv_cg1_artifacts", "upstream_scripts")
    }
    states_now = {
        "sgv15": list(
            cg1._decision_state(s15.DECISION, ("verdict", "criteria_met", "ready_for_sgv16"))
        ),
        "sgv15b": list(
            cg1._decision_state(s15b.DECISION, ("verdict", "criteria_met", "ready_for_sgv16"))
        ),
        "sgv_dt1": list(
            cg1._decision_state(
                dt1.DECISION, ("verdict", "criteria_met", "ready_for_external_confirmation")
            )
        ),
    }
    decisions_unchanged = states_now == freeze["upstream_decisions"]
    existing = [path for path in PRODUCED if path.exists()]
    raw = sorted(RAW_CACHE.glob("*.parquet"))
    audit = audit_report(REPORT, REPORT_SECTIONS) if REPORT.is_file() else None
    decision = cc_read_json(DECISION)
    _write_json_once(
        PROVENANCE,
        {
            **_envelope("provenance"),
            "issued_head": _git("rev-parse", "HEAD"),
            "gen1_starting_commit": freeze["gen1_starting_commit"],
            "working_tree_dirty": bool(_git("status", "--porcelain")),
            "artifacts": {_relative(p): file_sha256(p) for p in existing},
            "artifact_count": len(existing),
            "figures": {
                k: v["sha256"] for k, v in cc_read_json(FIGURE_MANIFEST)["figures"].items()
            },
            "raw_generation_outputs": {_relative(p): file_sha256(p) for p in raw},
            "raw_generation_output_count": len(raw),
            "report": {_relative(REPORT): file_sha256(REPORT) if REPORT.is_file() else None},
            "upstream_unchanged_since_section_zero": {
                **{f"{group}_changed": paths for group, paths in changed.items()},
                "upstream_decisions_unchanged": decisions_unchanged,
            },
            "decision_rederived_identically": rederived,
            "dependency_audit": {
                "artifacts_tracked_by_git": [_relative(p) for p in existing if _tracked(p)],
                "artifacts_not_git_ignored": [_relative(p) for p in existing if not _ignored(p)],
                "raw_data_written": False,
                "upstream_artifacts_written": any(changed.values()),
                "confirmatory_reserve_consumed": False,
                "external_confirmation_run": False,
                "reliability_transfer_started": False,
                "models_fine_tuned": 0,
                "external_api_calls": 0,
                "ground_truth_used_for": [
                    "known-site localization (location only, never content)",
                    "the population and sample of evaluable sites",
                    "outcome labelling and exact-repair determination",
                    "the marker-shuffle decoy position",
                    "the GT-corrector control",
                    "stratification and the analysis-only diagnostics",
                ],
                "deployable": decision["deployable"],
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
            "audited_classes": ["integers", "grouped integers", "decimals", "percentages"],
            "sections": {n: [_relative(p) for p in paths] for n, paths in REPORT_SECTIONS.items()},
            "audit": (
                {k: audit[k] for k in ("numeric_claims", "by_class", "untraceable_numeric_claims")}
                if audit
                else None
            ),
            "untraceable": audit["untraceable"] if audit else None,
        },
    )
    if any(changed.values()) or not decisions_unchanged:
        raise PhaseError("an upstream artifact changed since section 0; see provenance.json")
    if audit is None or audit["untraceable_numeric_claims"]:
        raise PhaseError(
            f"report traceability: {audit['untraceable_numeric_claims'] if audit else 'no report'}"
        )
    claims = audit["numeric_claims"]
    print(f"record: {len(existing)} artifacts, {len(raw)} raw shards, report claims {claims}")
    return 0


PHASES: tuple[tuple[str, Callable[[], int]], ...] = (
    ("reconstruct", run_reconstruct),
    ("freeze", run_freeze),
    ("preregister", run_preregister),
    ("probe", run_probe),
    ("audit", run_audit),
    ("availability", run_availability),
    ("sites", run_sites),
    ("contexts", run_contexts),
    ("registry", run_registry),
    ("generate", run_generate),
    ("spotcheck", run_spotcheck),
    ("normalize", run_normalize),
    ("reproduce", run_reproduce),
    ("exact", run_exact),
    ("strata", run_strata),
    ("gains", run_gains),
    ("diagnostics", run_diagnostics),
    ("overlap", run_overlap),
    ("cost", run_cost),
    ("controls", run_controls),
    ("negative", run_negative),
    ("stats", run_stats),
    ("figures", run_figures),
    ("decide", run_decide),
    ("determinism", run_determinism),
    ("record", run_record),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    for name, _ in PHASES:
        parser.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    selected = [name for name, _ in PHASES if getattr(args, name)]
    if not selected:
        parser.print_help()
        return 2
    lookup = dict(PHASES)
    for name in selected:
        began = time.monotonic()
        try:
            status = lookup[name]()
        except PhaseError as error:
            print(f"{name}: FAILED -- {error}", file=sys.stderr)
            return 1
        if status:
            return status
        print(f"  [{name} {time.monotonic() - began:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
