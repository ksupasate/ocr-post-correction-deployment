#!/usr/bin/env python3
"""SGV1 development phase 2: the image-representation ladder.

Session 2 answered "can this experiment be run and read honestly?" (C3_PASS) with a
deliberately weak image channel -- eight cheap statistics over the crop. It found a
positive but interval-crossing-zero effect on discrimination and calibration, and an exact
null on the matched-pair endpoint, where the correct crop and a crop from an unrelated
document produced the *same* delta. That is the finding this stage exists to interrogate.

The question here is not "can we get a higher AUC?" but::

    does the correct local source crop carry incremental information about whether
    THIS proposed edit is beneficial, over and above the strongest non-image verifier?

which is only answered by beating the irrelevant-image controls, not by beating V1. So
every rung runs five arms::

    correct       the crop this candidate's own OCR geometry names
    shuffled      a real crop from a DIFFERENT document          (relevance control)
    wrong_local   a real crop from the SAME document, other site (locality control)
    masked        the architecture with a structurally empty image channel (capacity)
    randproj      the same dimensionality filled with deterministic noise (capacity)

and the contrast that matters is ``correct - shuffled`` (and the stricter
``correct - wrong_local``), never ``correct - V1`` alone.

Fairness is structural rather than promised. Every rung is the SAME logistic regression,
on the SAME rows, in the SAME roles, with the SAME calibrator, over ``[V1's 47 columns ||
I_k]``. Encoders are fitted transformers: trained on TRAIN documents only, frozen, then
applied everywhere. Nothing but ``I_k`` changes between V1 and V2_k.

    --bind-baseline  hash-bind the Session-2 Rung A result so it can never be rewritten
    --register       declare the finite ladder, crop grid, seeds, and selection rule
    --crops          materialize the crop-scale grid and the extended donor lineage
    --encode         fit encoders on TRAIN and write frozen embeddings
    --fit            fit / calibrate / evaluate every rung x arm
    --analyze        core+all frame metrics, locality contrasts, per-engine, clustered CIs
    --figures        figures 1-9
    --decide         the machine-readable representation decision

This stage is DEVELOPMENT. It may not touch the confirmatory reserve, may not freeze a
final architecture, and may not issue an SGV1-H1 verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sgv1_verifier_pilot as pilot
from ocr_risk.calibrate.calibrators import build_calibrator
from ocr_risk.evidence.crops import CropPolicy, build_recipe, materialize
from ocr_risk.evidence.mask import EvidenceMask
from ocr_risk.io.hashing import file_sha256
from ocr_risk.io.paths import cache_root
from ocr_risk.metrics.calibration import brier_score, expected_calibration_error
from ocr_risk.metrics.discrimination import roc_auc
from ocr_risk.schemas.base import BBox
from ocr_risk.stats.bootstrap import paired_cluster_bootstrap

REPO = pilot.REPO
REPR_DIR = REPO / "results/generated/sgv1/representation"
BASELINE_RECORD = REPR_DIR / "baseline_v2a.json"
REGISTRY = REPR_DIR / "representation_registry.json"
CROP_SCALE_REGISTRY = REPR_DIR / "crop_scale_registry.json"
LINEAGE = REPR_DIR / "representation_lineage.parquet"
ENCODER_DIR = REPR_DIR / "encoders"
MODEL_DIR = REPR_DIR / "models"
FIGURE_DIR = REPR_DIR / "figures"
SCORES = REPR_DIR / "representation_scores.parquet"
FIT_RECORD = REPR_DIR / "representation_record.json"
METRICS = REPR_DIR / "representation_metrics.json"
COMPARISON = REPR_DIR / "representation_comparison.csv"
FIGURE_MANIFEST = REPR_DIR / "figure_manifest.json"
DECISION = REPR_DIR / "representation_decision.json"

# ---------------------------------------------------------------- preregistration
#
# Declared BEFORE any representation metric is computed and written to an immutable
# registry, because "we kept adding encoders until one worked" and "we tested three
# encoders" are indistinguishable after the fact unless the set was fixed in advance.

EMBED_DIM = 32
"""Deliberately small. A 512-d embedding appended to 47 non-image columns would make the
capacity difference between V1 and V2 the dominant change rather than the image channel."""

RUNGS: tuple[str, ...] = ("A", "B", "C")
IMAGE_ARMS: tuple[str, ...] = ("correct", "shuffled", "wrong_local", "masked", "randproj")

CROP_SCALES: dict[str, CropPolicy] = {
    # local_medium is byte-for-byte the Session-2 policy, so Rung A stays comparable.
    "local_tight": CropPolicy(padding_ratio=0.10, padding_min_px=2, target_height=48),
    "local_medium": CropPolicy(padding_ratio=0.25, padding_min_px=4, target_height=48),
    "local_contextual": CropPolicy(padding_ratio=1.00, padding_min_px=8, target_height=64),
}
BASELINE_SCALE = "local_medium"

ENCODER_SEEDS: tuple[int, ...] = (20260831, 20260901, 20260902)
PRIMARY_SEED = ENCODER_SEEDS[0]
"""The primary result is the primary seed. The other two are stability evidence, never a
menu to choose the best from."""

WRONG_LOCAL_SEED = 20260902
RANDPROJ_SEED = 20260903

# Gate 2 (locality). A non-trivial positive effect is required; statistical significance
# is NOT required at development stage, but a null or negative locality effect is a fail.
LOCALITY_MIN_AUC = 0.005
# Gate 4 (engine robustness).
ENGINE_REGRESSION_MAX = 0.02

INPUT_HEIGHT = 64
INPUT_WIDTH = 256
BATCH_SIZE = 128
MAX_EPOCHS = 12
PATIENCE = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
INNER_DEV_FRACTION = 0.15

RESNET18_CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
RESNET18_PROVENANCE = {
    "model": "torchvision.models.resnet18",
    "weights_enum": "ResNet18_Weights.IMAGENET1K_V1",
    "source_url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "licence": "BSD-3-Clause (torchvision)",
    "pretraining_corpus": "ImageNet-1k",
    "limitation": (
        "A natural-image classifier, not a document or scene-text encoder. It is what is "
        "reproducibly available offline in this environment; a document-pretrained "
        "encoder would be a stronger rung C and is recorded as a gap, not claimed."
    ),
}


class RepresentationError(RuntimeError):
    """A freeze, fairness, or preregistration invariant failed."""


# ------------------------------------------------------------------------- io


def _relative(path: Path) -> str:
    return pilot._relative(path)


def _read_json(path: Path) -> dict[str, Any]:
    return pilot._read_json(path)


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    pilot._write_json_once(path, payload)


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    pilot._write_parquet_once(path, frame)


def _digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _policy_payload(policy: CropPolicy) -> dict[str, Any]:
    return {
        "padding_ratio": policy.padding_ratio,
        "padding_min_px": policy.padding_min_px,
        "target_height": policy.target_height,
        "grayscale": policy.grayscale,
    }


# ------------------------------------------------------- stage 1: bind rung A


def run_bind_baseline() -> int:
    """Hash-bind the Session-2 result so representation work cannot rewrite history.

    Rung A is closed. Its scores, crops, feature configuration, model configuration and
    metrics are recorded here by content hash; every later stage re-checks them, so a
    later edit to the pilot that changed V1 would surface as a broken binding rather than
    as a quietly improved baseline.
    """
    started = time.monotonic()
    pilot_record = _read_json(pilot.PILOT_RECORD)
    evidence_record = _read_json(pilot.EVIDENCE_RECORD)
    c3 = _read_json(pilot.C3_CERTIFICATE)

    for path, recorded in (
        (pilot.SCORES_TABLE, pilot_record["artifacts"][_relative(pilot.SCORES_TABLE)]),
        (pilot.EVIDENCE_TABLE, evidence_record["artifacts"][_relative(pilot.EVIDENCE_TABLE)]),
        (pilot.CROP_LINEAGE, evidence_record["artifacts"][_relative(pilot.CROP_LINEAGE)]),
    ):
        if file_sha256(path) != recorded:
            raise RepresentationError(f"{_relative(path)} has moved since its record")

    scores = pd.read_parquet(pilot.SCORES_TABLE)
    arms = ("V0", "V1", "V2_correct", "V2_shuffled", "V2_masked")
    score_hashes = {
        arm: _digest(*(f"{v:.17g}" for v in scores[f"score_{arm}"].to_numpy())) for arm in arms
    }
    prob_hashes = {
        arm: _digest(*(f"{v:.17g}" for v in scores[f"pharm_{arm}"].to_numpy())) for arm in arms
    }

    record = {
        "schema_version": "sgv1-representation-baseline-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "rung": "A",
        "representation": "image_block: 8 cheap statistics over the crop",
        "implementation": "src/ocr_risk/verify/featurizers.py::image_block",
        "implementation_sha256": file_sha256(REPO / "src/ocr_risk/verify/featurizers.py"),
        "status": "CLOSED -- historical baseline, never retuned in this phase",
        "crop_policy": _policy_payload(pilot.CROP_POLICY),
        "crop_scale_name": BASELINE_SCALE,
        "feature_configuration": {
            arm: {
                "evidence_config": pilot_record["arms"][arm]["evidence_config"],
                "evidence_fields": pilot_record["arms"][arm]["evidence_fields"],
                "feature_dimension": pilot_record["arms"][arm]["feature_dimension"],
                "feature_names_sha256": _digest(*pilot_record["arms"][arm]["feature_names"]),
                "crop_mode": pilot_record["arms"][arm]["crop_mode"],
                "provenance_block": pilot_record["arms"][arm]["provenance_block"],
            }
            for arm in arms
        },
        "model_configuration": {
            arm: {
                "model": pilot_record["arms"][arm]["model"],
                "C": pilot_record["arms"][arm]["C"],
                "max_iter": pilot_record["arms"][arm]["max_iter"],
                "class_weight": pilot_record["arms"][arm]["class_weight"],
                "random_state": pilot_record["arms"][arm]["random_state"],
                "calibration_method": pilot_record["arms"][arm]["calibration_method"],
                "calibrator_identity": pilot_record["arms"][arm]["calibrator_identity"],
                "model_parameters": pilot_record["arms"][arm]["model_parameters"],
                "fit_rows": pilot_record["arms"][arm]["fit_rows"],
                "calibration_rows": pilot_record["arms"][arm]["calibration_rows"],
                "evaluation_rows": pilot_record["arms"][arm]["evaluation_rows"],
            }
            for arm in arms
        },
        "score_column_sha256": score_hashes,
        "calibrated_probability_sha256": prob_hashes,
        "metrics": pilot_record["metrics"],
        "artifacts": {
            _relative(pilot.SCORES_TABLE): file_sha256(pilot.SCORES_TABLE),
            _relative(pilot.CROP_LINEAGE): file_sha256(pilot.CROP_LINEAGE),
            _relative(pilot.EVIDENCE_TABLE): file_sha256(pilot.EVIDENCE_TABLE),
            _relative(pilot.PILOT_RECORD): file_sha256(pilot.PILOT_RECORD),
            _relative(pilot.EVIDENCE_RECORD): file_sha256(pilot.EVIDENCE_RECORD),
            _relative(pilot.C3_CERTIFICATE): file_sha256(pilot.C3_CERTIFICATE),
            _relative(pilot.STATISTICS_RECORD): file_sha256(pilot.STATISTICS_RECORD),
        },
        "c3_verdict": c3["verdict"],
        "c2_status": c3["c2_status"],
        "confirmatory_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(BASELINE_RECORD, record)
    print(f"baseline bound: rung A, {len(arms)} arms -> {_relative(BASELINE_RECORD)}")
    return 0


def _assert_baseline_intact() -> dict[str, Any]:
    """Every later stage re-checks the Rung A binding before adding to the ladder."""
    record = _read_json(BASELINE_RECORD)
    for relative, recorded in record["artifacts"].items():
        actual = file_sha256(REPO / relative)
        if actual != recorded:
            raise RepresentationError(
                f"rung A baseline moved: {relative} is {actual[:12]}, bound as {recorded[:12]}"
            )
    return record


# --------------------------------------------------- stage 2: preregistration


def run_register() -> int:
    """Freeze the finite ladder, crop grid, seed set, and selection rule."""
    started = time.monotonic()
    baseline = _assert_baseline_intact()
    registry = {
        "schema_version": "sgv1-representation-registry-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "stage": "DEVELOPMENT",
        "declared_before_any_representation_metric": True,
        "statement": (
            "This registry fixes the search space before results are seen. Adding a rung, "
            "a crop scale, or a seed after this point requires a new registry version and "
            "must be reported as an extension of the family, not as part of it."
        ),
        "representations": {
            "A": {
                "name": "crop statistics",
                "kind": "handcrafted",
                "dimension": 8,
                "status": "closed historical baseline",
                "trained": False,
            },
            "B": {
                "name": "compact convolutional local encoder",
                "kind": "learned",
                "dimension": EMBED_DIM + 1,
                "status": "to be trained on TRAIN documents only",
                "trained": True,
            },
            "C": {
                "name": "frozen ImageNet ResNet-18 backbone with a trained projection head",
                "kind": "pretrained frozen backbone + learned projection",
                "dimension": EMBED_DIM + 1,
                "status": "to be fitted on TRAIN documents only",
                "trained": True,
                "backbone_provenance": RESNET18_PROVENANCE,
            },
            "D": {
                "name": "small multimodal verifier",
                "status": "NOT RUN -- optional rung, gated on B/C being ambiguous",
                "trained": False,
            },
        },
        "image_arms": list(IMAGE_ARMS),
        "arm_semantics": {
            "correct": "the crop this candidate's own OCR anchor geometry names",
            "shuffled": "a real crop from a different document (relevance control)",
            "wrong_local": "a real crop from the same document at a different site "
            "(locality control: separates document appearance from site evidence)",
            "masked": "the same feature space, structurally empty (capacity control)",
            "randproj": "the same dimensionality filled with deterministic noise "
            "(capacity control: can extra dimensions alone be exploited?)",
        },
        "crop_scales": {name: _policy_payload(p) for name, p in CROP_SCALES.items()},
        "baseline_scale": BASELINE_SCALE,
        "crop_scale_rule": (
            "A finite three-point grid declared here. Crop scale is selected once, on "
            "DEVELOPMENT data, by the Gate 2/3 rule below, and that selection is recorded "
            "as a development decision. There is no continuous tuning."
        ),
        "encoder_seeds": list(ENCODER_SEEDS),
        "primary_seed": PRIMARY_SEED,
        "seed_rule": (
            "The primary result is the primary seed. The remaining seeds are reported as "
            "stability evidence and may not be selected among."
        ),
        "primary_frame": "frame_a_core",
        "primary_frame_reason": (
            "Incident L4: at a whitespace-only gap region overcorrection is structurally "
            "unreachable, so 699 of Frame A's 1,320 harmful rows are separable on a "
            "non-image structural cue. The core slice is primary; the all-frame result is "
            "sensitivity analysis. Gap rows are neither deleted nor relabelled."
        ),
        "primary_endpoint": "core Frame-A ROC AUC",
        "secondary_endpoints": [
            "core Frame-A PR AUC",
            "matched-pair P(q+ > q-)",
            "Brier score",
            "mean calibrated harm probability on clean-anchor overcorrection rows",
        ],
        "central_contrast": "correct minus shuffled (locality), not correct minus V1",
        "selection_rule": {
            "gate_1_validity": [
                "rung A baseline binding intact",
                "every arm scores exactly the V1 candidate id set, in order",
                "non-image columns identical to V1 by construction",
                "crop lineage geometry re-derives from frozen OCR spans",
                "masked arm reproduces V1 within 1e-9",
                "no confirmatory document in any role",
            ],
            "gate_2_locality": {
                "requirement": "core Frame-A ROC AUC(correct) - ROC AUC(shuffled) > "
                f"{LOCALITY_MIN_AUC}",
                "also_required": "core AUC(correct) > core AUC(wrong_local)",
                "significance_required": False,
                "note": "A non-trivial positive locality effect is required. Statistical "
                "significance is not required at development stage.",
            },
            "gate_3_incremental": [
                "core Frame-A ROC AUC delta versus V1",
                "core Frame-A PR AUC delta versus V1",
                "Brier improvement versus V1",
                "matched-pair ranking delta versus V1",
            ],
            "gate_4_engine_robustness": {
                "requirement": "no engine may lose more than "
                f"{ENGINE_REGRESSION_MAX} ROC AUC versus V1",
                "note": "A rung that helps one engine and substantially harms another is "
                "characterized as engine-dependent, never presented as a pooled win.",
            },
        },
        "decision_statuses": [
            "REPRESENTATION_READY",
            "REPRESENTATION_PARTIALLY_READY",
            "REPRESENTATION_NOT_READY",
            "REPRESENTATION_INCONCLUSIVE",
        ],
        "prohibited_in_this_stage": [
            "confirmatory reserve access",
            "final architecture freeze",
            "final primary endpoint freeze",
            "risk-UCB selection",
            "C2 decision",
            "any SGV1-H1 verdict",
        ],
        "baseline_binding_sha256": file_sha256(BASELINE_RECORD),
        "baseline_c3_verdict": baseline["c3_verdict"],
        "confirmatory_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(REGISTRY, registry)
    print(
        f"registered: {len(RUNGS)} rungs x {len(CROP_SCALES)} scales x {len(IMAGE_ARMS)} arms "
        f"-> {_relative(REGISTRY)}"
    )
    return 0


def _assert_registered() -> dict[str, Any]:
    if not REGISTRY.is_file():
        raise RepresentationError("run --register before any representation metric")
    registry = _read_json(REGISTRY)
    if registry["baseline_binding_sha256"] != file_sha256(BASELINE_RECORD):
        raise RepresentationError("the rung A binding moved after registration")
    declared = {name: _policy_payload(p) for name, p in CROP_SCALES.items()}
    if registry["crop_scales"] != declared:
        raise RepresentationError("the crop-scale grid differs from the registered grid")
    if list(registry["image_arms"]) != list(IMAGE_ARMS):
        raise RepresentationError("the arm set differs from the registered set")
    if list(registry["encoder_seeds"]) != list(ENCODER_SEEDS):
        raise RepresentationError("the seed set differs from the registered set")
    return registry


# -------------------------------------------------- stage 3: the crop grid


def _bbox_of(row: Any) -> BBox:
    return BBox(
        x0=float(row.bbox_x0), y0=float(row.bbox_y0), x1=float(row.bbox_x1), y1=float(row.bbox_y1)
    )


def _pick(seed: int, candidate_id: str, options: Sequence[str]) -> str:
    digest = hashlib.sha256(f"{seed}:{candidate_id}".encode()).digest()
    return options[int.from_bytes(digest[:8], "big") % len(options)]


def run_crops() -> int:
    """Materialize the declared crop-scale grid and the extended donor lineage.

    Adds one donor the Session-2 pilot did not have: ``wrong_local``, a real crop from the
    SAME page and the SAME engine at a DIFFERENT site. The shuffled arm can be beaten by
    recognizing generic document appearance -- paper, typeface, scanner -- because the
    donor page is a different receipt. ``wrong_local`` holds all of that constant and
    varies only *where on this page* the verifier is looking, which is the thing the
    source-grounding claim is actually about.
    """
    started = time.monotonic()
    _assert_registered()
    _assert_baseline_intact()

    base = pd.read_parquet(pilot.CROP_LINEAGE)
    documents = sorted(base["document_id"].astype(str).unique())
    images = pilot._page_images(set(documents))

    frame = base[
        [
            "candidate_id",
            "site_id",
            "document_id",
            "engine_id",
            "role",
            "source_image_sha256",
            "bbox_x0",
            "bbox_y0",
            "bbox_x1",
            "bbox_y1",
            "crop_recipe_sha256",
            "shuffled_donor_document_id",
        ]
    ].copy()

    scale_stats: dict[str, Any] = {}
    for scale, policy in CROP_SCALES.items():
        recipes: list[str] = []
        seen: dict[str, Any] = {}
        for row in frame.itertuples():
            recipe = build_recipe(str(row.source_image_sha256), _bbox_of(row), policy)
            recipes.append(recipe.recipe_sha256)
            if recipe.recipe_sha256 not in seen:
                seen[recipe.recipe_sha256] = recipe
                materialize(recipe, images[str(row.document_id)].path)
        frame[f"recipe_{scale}"] = recipes

        # local_medium must reproduce the Session-2 recipes exactly, or Rung A is not
        # comparable to the new rungs and the whole ladder is measuring two things.
        if scale == BASELINE_SCALE:
            mismatch = int((frame[f"recipe_{scale}"] != frame["crop_recipe_sha256"]).sum())
            if mismatch:
                raise RepresentationError(
                    f"{mismatch} rows: {BASELINE_SCALE} does not reproduce the Session-2 recipes"
                )

        by_document: dict[str, list[str]] = {}
        by_document_engine: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for row in frame.itertuples():
            digest = str(getattr(row, f"recipe_{scale}"))
            by_document.setdefault(str(row.document_id), []).append(digest)
            key = (str(row.document_id), str(row.engine_id))
            by_document_engine.setdefault(key, []).append((str(row.site_id), digest))
        for values in by_document.values():
            values.sort()
        for pairs_list in by_document_engine.values():
            pairs_list.sort()

        shuffled: list[str | None] = []
        wrong_local: list[str | None] = []
        for row in frame.itertuples():
            candidate_id = str(row.candidate_id)
            donor_pool = by_document.get(str(row.shuffled_donor_document_id))
            shuffled.append(
                _pick(pilot.SHUFFLE_SEED, f"donor:{candidate_id}", donor_pool)
                if donor_pool
                else None
            )
            same_page = by_document_engine.get((str(row.document_id), str(row.engine_id)), [])
            # A different site is not sufficient. Distinct sites can share an anchor span
            # union -- a token site and a token-pair site over the same span, say -- and
            # then the "wrong" crop is byte-identical to the right one. Excluding by
            # recipe as well as by site is what makes this arm a control rather than a
            # second copy of the correct arm; it cost 222 rows to discover.
            own = str(getattr(row, f"recipe_{scale}"))
            elsewhere = sorted({d for s, d in same_page if s != str(row.site_id) and d != own})
            wrong_local.append(
                _pick(WRONG_LOCAL_SEED, candidate_id, elsewhere) if elsewhere else None
            )
        frame[f"shuffled_{scale}"] = shuffled
        frame[f"wrong_local_{scale}"] = wrong_local

        same_page_leak = int(
            (frame[f"wrong_local_{scale}"] == frame[f"recipe_{scale}"]).fillna(False).sum()
        )
        if same_page_leak:
            raise RepresentationError(
                f"{same_page_leak} wrong_local donors are the candidate's own crop"
            )
        scale_stats[scale] = {
            "policy": _policy_payload(policy),
            "unique_recipes": int(frame[f"recipe_{scale}"].nunique()),
            "crops_materialized": len(seen),
            "shuffled_available": int(frame[f"shuffled_{scale}"].notna().sum()),
            "wrong_local_available": int(frame[f"wrong_local_{scale}"].notna().sum()),
            "wrong_local_unavailable": int(frame[f"wrong_local_{scale}"].isna().sum()),
            "reproduces_session_2_recipes": scale == BASELINE_SCALE,
        }
        print(f"  {scale}: {len(seen)} unique crops materialized")

    _write_parquet_once(LINEAGE, frame)
    record = {
        "schema_version": "sgv1-crop-scale-registry-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "rows": len(frame),
        "documents": len(documents),
        "scales": scale_stats,
        "geometry_source": "ocr_span_bbox_union (inherited verbatim from the Session-2 lineage)",
        "geometry_never_from_ground_truth": True,
        "donor_rules": {
            "shuffled": "the Session-2 donor DOCUMENT, then a real recipe of that document "
            "at this scale, chosen by sha256(SHUFFLE_SEED:donor:candidate_id)",
            "wrong_local": "a real recipe from the SAME document and SAME engine at a "
            "DIFFERENT site, chosen by sha256(WRONG_LOCAL_SEED:candidate_id)",
        },
        "wrong_local_seed": WRONG_LOCAL_SEED,
        "shuffle_seed": pilot.SHUFFLE_SEED,
        "inputs": {_relative(pilot.CROP_LINEAGE): file_sha256(pilot.CROP_LINEAGE)},
        "artifacts": {_relative(LINEAGE): file_sha256(LINEAGE)},
        "confirmatory_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(CROP_SCALE_REGISTRY, record)
    print(f"crops: {len(frame)} rows x {len(CROP_SCALES)} scales -> {_relative(LINEAGE)}")
    return 0


# ------------------------------------------------------ stage 4: the encoders


def _decode(digest: str) -> np.ndarray:
    """One crop as a fixed-size grayscale array, aspect preserved then padded.

    Padding rather than stretching: a squashed 400px receipt line and a squashed 40px one
    would look alike to the encoder, and stroke geometry is precisely what a source-
    grounded verifier is supposed to read.
    """
    path = cache_root() / "crops" / digest[:2] / f"{digest}.png"
    canvas = np.full((INPUT_HEIGHT, INPUT_WIDTH), 255, dtype=np.uint8)
    if not path.is_file():
        return canvas
    with Image.open(path) as opened:
        image = opened.convert("L")
        scale = INPUT_HEIGHT / max(image.height, 1)
        width = max(1, min(INPUT_WIDTH, round(image.width * scale)))
        image = image.resize((width, INPUT_HEIGHT), resample=Image.Resampling.LANCZOS)
        canvas[:, :width] = np.asarray(image, dtype=np.uint8)
    return canvas


def _decode_universe(digests: Sequence[str]) -> tuple[dict[str, int], np.ndarray]:
    """Decode each distinct crop once; every arm then indexes into the same array."""
    unique = sorted(set(digests))
    index = {digest: i for i, digest in enumerate(unique)}
    store = np.empty((len(unique), INPUT_HEIGHT, INPUT_WIDTH), dtype=np.uint8)
    for digest, position in index.items():
        store[position] = _decode(digest)
    return index, store


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment guard
        raise RepresentationError("rungs B and C need torch; install the torch extra") from error
    return torch


def _device(torch: Any) -> Any:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_encoder(rung: str, torch: Any, seed: int) -> Any:
    """Rung B trains its own convolutional trunk; rung C freezes an ImageNet backbone."""
    from torch import nn

    torch.manual_seed(seed)

    class ConvLocalEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 4)),
            )
            self.project = nn.Linear(64 * 4, EMBED_DIM)
            self.head = nn.Linear(EMBED_DIM, 1)

        def embed(self, x: Any) -> Any:
            return self.project(self.trunk(x).flatten(1))

        def forward(self, x: Any) -> Any:
            return self.head(self.embed(x)).squeeze(1)

    class FrozenResNetEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            from torchvision.models import resnet18

            if not RESNET18_CHECKPOINT.is_file():
                raise RepresentationError(
                    f"rung C needs the local checkpoint {RESNET18_CHECKPOINT}; "
                    "this stage does not download models"
                )
            backbone = resnet18(weights=None)
            backbone.load_state_dict(torch.load(RESNET18_CHECKPOINT, map_location="cpu"))
            backbone.fc = nn.Identity()
            for parameter in backbone.parameters():
                parameter.requires_grad = False
            backbone.eval()
            self.backbone = backbone
            self.project = nn.Linear(512, EMBED_DIM)
            self.head = nn.Linear(EMBED_DIM, 1)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def embed(self, x: Any) -> Any:
            # The backbone stays in eval mode for good: its BatchNorm running statistics
            # are ImageNet's, and letting them drift on TRAIN crops would make the
            # "frozen pretrained backbone" label false.
            self.backbone.eval()
            with torch.no_grad():
                rgb = x.repeat(1, 3, 1, 1)
                rgb = (rgb - self.mean.to(x.device)) / self.std.to(x.device)
                features = self.backbone(rgb)
            return self.project(features)

        def forward(self, x: Any) -> Any:
            return self.head(self.embed(x)).squeeze(1)

    return ConvLocalEncoder() if rung == "B" else FrozenResNetEncoder()


def _inner_split(documents: Sequence[str], seed: int) -> set[str]:
    """A document-disjoint inner development split carved out of TRAIN.

    Early stopping needs a held-out signal, and taking it from CALIBRATION or DEVELOPMENT
    would be selection on the evaluation data. It comes out of TRAIN, by document.
    """
    ordered = sorted(set(documents))
    picked = {
        d
        for d in ordered
        if int.from_bytes(hashlib.sha256(f"{seed}:inner:{d}".encode()).digest()[:8], "big") % 10_000
        < INNER_DEV_FRACTION * 10_000
    }
    return picked or {ordered[0]}


@dataclass(frozen=True, slots=True)
class TrainingReport:
    epochs_run: int
    best_epoch: int
    best_inner_auc: float
    inner_documents: int
    inner_rows: int
    train_rows: int
    parameters: int
    trainable_parameters: int
    device: str
    runtime_seconds: float
    checkpoint_sha256: str


def _train_encoder(
    rung: str,
    scale: str,
    seed: int,
    frame: pd.DataFrame,
    index: dict[str, int],
    store: np.ndarray,
) -> tuple[Any, TrainingReport]:
    """Fit the encoder on TRAIN correct-crops only, then freeze it.

    The encoder is a fitted transformer in exactly the sense the repository's leakage rules
    use: its parameters come from the fit role and are frozen before any calibration or
    development row is touched. The classifier on top is still the same logistic
    regression every other arm uses.
    """
    torch = _torch()
    from torch import nn

    started = time.monotonic()
    device = _device(torch)
    train = frame[frame["role"] == "TRAIN"].reset_index(drop=True)
    inner_documents = _inner_split(list(train["document_id"].astype(str)), seed)
    is_inner = train["document_id"].astype(str).isin(inner_documents).to_numpy()

    rows = np.asarray([index[d] for d in train[f"recipe_{scale}"].astype(str)])
    target = train["is_harmful"].to_numpy(dtype=np.float32)
    outer, inner = rows[~is_inner], rows[is_inner]
    y_outer, y_inner = target[~is_inner], target[is_inner]

    model = _build_encoder(rung, torch, seed).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    positives = float(y_outer.sum())
    pos_weight = torch.tensor(
        [(len(y_outer) - positives) / max(positives, 1.0)], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = np.random.default_rng(seed)

    best_auc, best_epoch, best_state, epochs_run = -np.inf, 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        if rung == "C":
            model.backbone.eval()
        order = generator.permutation(len(outer))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            x = torch.from_numpy(store[outer[batch]]).to(device).float().div_(255.0).unsqueeze(1)
            y = torch.from_numpy(y_outer[batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        epochs_run = epoch

        model.eval()
        with torch.no_grad():
            logits = np.concatenate(
                [
                    model(
                        torch.from_numpy(store[inner[s : s + 512]])
                        .to(device)
                        .float()
                        .div_(255.0)
                        .unsqueeze(1)
                    )
                    .cpu()
                    .numpy()
                    for s in range(0, len(inner), 512)
                ]
            )
        auc = roc_auc(logits.astype(np.float64), y_inner.astype(np.float64))
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL_DIR / f"rung{rung}_{scale}_seed{seed}.pt"
    torch.save(model.state_dict(), checkpoint)
    report = TrainingReport(
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        best_inner_auc=float(best_auc),
        inner_documents=len(inner_documents),
        inner_rows=int(is_inner.sum()),
        train_rows=int((~is_inner).sum()),
        parameters=int(sum(p.numel() for p in model.parameters())),
        trainable_parameters=int(sum(p.numel() for p in trainable)),
        device=str(device),
        runtime_seconds=time.monotonic() - started,
        checkpoint_sha256=file_sha256(checkpoint),
    )
    return model, report


def _embed_all(
    model: Any, digests: Sequence[str], index: dict[str, int], store: np.ndarray
) -> np.ndarray:
    torch = _torch()
    device = _device(torch)
    rows = np.asarray([index.get(d, -1) for d in digests])
    output = np.zeros((len(rows), EMBED_DIM + 1), dtype=np.float64)
    present = rows >= 0
    output[~present, EMBED_DIM] = 1.0
    positions = rows[present]
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(positions), 512):
            batch = store[positions[start : start + 512]]
            x = torch.from_numpy(batch).to(device).float().div_(255.0).unsqueeze(1)
            chunks.append(model.embed(x).cpu().numpy().astype(np.float64))
    if chunks:
        output[present, :EMBED_DIM] = np.vstack(chunks)
    return output


def _embedding_path(rung: str, scale: str, seed: int, arm: str) -> Path:
    return cache_root() / "sgv1_repr" / f"{rung}_{scale}_{seed}_{arm}.parquet"


ENCODED_ARMS: tuple[str, ...] = ("correct", "shuffled", "wrong_local")


def _encode_plan(rung: str | None, scale: str | None) -> list[tuple[str, str, int]]:
    plan: list[tuple[str, str, int]] = []
    for r in ("B", "C"):
        for s in CROP_SCALES:
            plan.append((r, s, PRIMARY_SEED))
        if r == "B":
            plan.extend((r, BASELINE_SCALE, seed) for seed in ENCODER_SEEDS[1:])
    return [p for p in plan if (rung in (None, p[0])) and (scale in (None, p[1]))]


def run_encode(rung: str | None = None, scale: str | None = None) -> int:
    started = time.monotonic()
    _assert_registered()
    _assert_baseline_intact()
    if not LINEAGE.is_file():
        raise RepresentationError("run --crops before --encode")

    lineage = pd.read_parquet(LINEAGE)
    labels = pd.read_parquet(pilot.LABEL_TABLE)[["candidate_id", "is_harmful"]]
    frame = lineage.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    if len(frame) != len(lineage):
        raise RepresentationError("lineage and labels do not align one to one")

    reports: dict[str, Any] = {}
    for r, s, seed in _encode_plan(rung, scale):
        columns = [f"{arm}_{s}" if arm != "correct" else f"recipe_{s}" for arm in ENCODED_ARMS]
        digests = pd.unique(pd.concat([frame[c].dropna() for c in columns]).astype(str))
        index, store = _decode_universe(list(digests))
        model, report = _train_encoder(r, s, seed, frame, index, store)
        for arm, column in zip(ENCODED_ARMS, columns, strict=True):
            values = frame[column].astype("object").where(frame[column].notna(), "")
            matrix = _embed_all(model, [str(v) for v in values], index, store)
            path = _embedding_path(r, s, seed, arm)
            path.parent.mkdir(parents=True, exist_ok=True)
            table = pd.DataFrame(matrix, columns=list(EMBED_NAMES))
            table.insert(0, "candidate_id", frame["candidate_id"].astype(str).to_numpy())
            table.to_parquet(path, index=False)
        key = f"{r}@{s}#{seed}"
        reports[key] = {
            **asdict(report),
            "rung": r,
            "crop_scale": s,
            "seed": seed,
            "unique_crops_decoded": len(index),
            "embedding_dimension": EMBED_DIM + 1,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "early_stopping": "best inner-dev ROC AUC, patience 3, inner split carved "
            "from TRAIN documents only",
            "selection_scope": "TRAIN documents only (inner development split)",
            "input_shape": [1, INPUT_HEIGHT, INPUT_WIDTH],
            "preprocessing": "grayscale, aspect-preserving resize to height "
            f"{INPUT_HEIGHT}, right-padded to width {INPUT_WIDTH} with white, scaled to [0,1]",
            "augmentation": "none",
            "confirmatory_accessed": False,
        }
        print(
            f"  {key}: inner AUC {report.best_inner_auc:.4f} @epoch {report.best_epoch}, "
            f"{report.trainable_parameters} trainable params, {report.runtime_seconds:.0f}s"
        )

    # One write-once record per encoder rather than one accumulating file: --encode may be
    # invoked per rung, and rewriting a shared log to append would break the write-once
    # discipline every other artifact in this study is held to.
    for key, payload in reports.items():
        record = {
            "schema_version": "sgv1-representation-encoder-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": pilot._git_head(),
            "encoder": key,
            **payload,
            "backbone_provenance": RESNET18_PROVENANCE if payload["rung"] == "C" else None,
            "backbone_checkpoint_sha256": file_sha256(RESNET18_CHECKPOINT)
            if payload["rung"] == "C"
            else None,
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_json_once(ENCODER_DIR / f"{key.replace('@', '_').replace('#', '_')}.json", record)
    print(f"encoded: {len(reports)} encoders -> {_relative(ENCODER_DIR)}")
    return 0


# ---------------------------------------------------------- stage 5: fitting

EMBED_NAMES: tuple[str, ...] = (*(f"img_e{i:02d}" for i in range(EMBED_DIM)), "img_missing")


class RepresentationVerifier(pilot.SGV1Verifier):
    """V1's feature space plus one image-representation block, and nothing else.

    The image channel for rungs B and C arrives as a precomputed frozen embedding rather
    than through ``image_block``, so the evidence configuration stays ``sgv1_v1`` and the
    first 47 columns are V1's own columns *by construction* -- not by a promise that two
    code paths agree. ``--fit`` additionally re-fits V1 itself and asserts it reproduces
    the frozen Session-2 scores bit for bit, which is what makes that construction
    checkable rather than merely stated.
    """

    def __init__(self, *args: Any, embedding: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._embedding = embedding or {}

    def _matrix(self, inputs: Sequence[Any]) -> Any:
        base = super()._matrix(inputs)
        if not self._embedding:
            return base
        extra = np.vstack([self._embedding[item.candidate_id] for item in inputs])
        if len(self._feature_names) == base.shape[1]:
            self._feature_names = (*self._feature_names, *EMBED_NAMES)
        return np.hstack([base, extra])


@dataclass(frozen=True, slots=True)
class FitPlan:
    label: str
    rung: str | None
    scale: str | None
    seed: int | None
    arm: str


def _fit_plans() -> list[FitPlan]:
    plans = [
        FitPlan("V0", None, None, None, "none"),
        FitPlan("V1", None, None, None, "none"),
    ]
    for rung in ("B", "C"):
        for scale in CROP_SCALES:
            plans.extend(
                FitPlan(f"{rung}@{scale}#{PRIMARY_SEED}_{arm}", rung, scale, PRIMARY_SEED, arm)
                for arm in ENCODED_ARMS
            )
        plans.extend(
            FitPlan(
                f"{rung}@{BASELINE_SCALE}#{PRIMARY_SEED}_{arm}",
                rung,
                BASELINE_SCALE,
                PRIMARY_SEED,
                arm,
            )
            for arm in ("masked", "randproj")
        )
    for seed in ENCODER_SEEDS[1:]:
        plans.extend(
            FitPlan(f"B@{BASELINE_SCALE}#{seed}_{arm}", "B", BASELINE_SCALE, seed, arm)
            for arm in ENCODED_ARMS
        )
    return plans


def _randproj(candidate_ids: Sequence[str]) -> dict[str, Any]:
    """Deterministic noise of the real embedding's width.

    The capacity control the masked arm cannot be: masked columns are constant, so the
    scaler zeroes them and the model provably cannot use them. These columns carry real
    variance and no information, which is the harder test -- can the fitter exploit width
    alone?
    """
    out: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        seed = int.from_bytes(
            hashlib.sha256(f"{RANDPROJ_SEED}:{candidate_id}".encode()).digest()[:8], "big"
        )
        values = np.zeros(EMBED_DIM + 1, dtype=np.float64)
        values[:EMBED_DIM] = np.random.default_rng(seed).standard_normal(EMBED_DIM)
        out[candidate_id] = values
    return out


def _embedding_for(plan: FitPlan, candidate_ids: Sequence[str]) -> dict[str, Any] | None:
    if plan.rung is None:
        return None
    if plan.arm == "masked":
        blank = np.zeros(EMBED_DIM + 1, dtype=np.float64)
        blank[EMBED_DIM] = 1.0
        return dict.fromkeys(candidate_ids, blank)
    if plan.arm == "randproj":
        return _randproj(candidate_ids)
    assert plan.scale is not None and plan.seed is not None
    path = _embedding_path(plan.rung, plan.scale, plan.seed, plan.arm)
    if not path.is_file():
        raise RepresentationError(f"missing embedding {path.name}; run --encode first")
    table = pd.read_parquet(path)
    values = table[list(EMBED_NAMES)].to_numpy(dtype=np.float64)
    return dict(zip(table["candidate_id"].astype(str), values, strict=True))


def _pool() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """The Session-2 evaluation pool, rebuilt by the same rules the pilot used."""
    roles = pilot._locked_roles()
    candidates = pd.read_parquet(pilot.CANDIDATE_TABLE)
    labels = pd.read_parquet(pilot.LABEL_TABLE)
    pool = candidates.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    pool = pool[pool["labelable"]].reset_index(drop=True)
    manifest_role = pool["document_id"].map(roles)
    if manifest_role.isna().any():
        raise RepresentationError("a labelable candidate has no non-confirmatory role")
    if not bool((pool["role"].astype(str) == manifest_role.astype(str)).all()):
        raise RepresentationError("the frozen candidate role disagrees with the role manifest")
    pool["beneficial"] = pool["outcome"].isin({"true_correction", "partial_improvement"})
    bundles = pilot._load_bundles()
    site_counts = candidates.groupby("site_id").size().to_dict()
    provenance = {
        str(row.candidate_id): pilot.provenance_block(row, site_counts[row.site_id]).values
        for row in pool.itertuples()
    }
    return pool, bundles, provenance


def run_fit() -> int:
    started = time.monotonic()
    _assert_registered()
    _assert_baseline_intact()

    pool, bundles, provenance = _pool()
    lineage = pd.read_parquet(pilot.CROP_LINEAGE)
    fit_rows = pool[pool["role"] == "TRAIN"].reset_index(drop=True)
    cal_rows = pool[pool["role"] == "CALIBRATION"].reset_index(drop=True)
    dev_rows = pool[pool["role"] == "DEVELOPMENT"].reset_index(drop=True)
    if set(fit_rows["document_id"]) & set(dev_rows["document_id"]):
        raise RepresentationError("a document appears in both the fit and evaluation role")
    if set(cal_rows["document_id"]) & set(dev_rows["document_id"]):
        raise RepresentationError("a document appears in both the calibration and evaluation role")

    everything = list(pool["candidate_id"].astype(str))
    scores = dev_rows[
        [
            "candidate_id",
            "site_id",
            "document_id",
            "engine_id",
            "outcome",
            "is_harmful",
            "beneficial",
            "region_is_whitespace_only",
            "operation",
            "anchor_kind",
            "generator_source",
        ]
    ].copy()

    frozen = pd.read_parquet(pilot.SCORES_TABLE)
    if list(frozen["candidate_id"].astype(str)) != list(scores["candidate_id"].astype(str)):
        raise RepresentationError("the development row order differs from the Session-2 pilot")
    for arm in ("V2_correct", "V2_shuffled", "V2_masked"):
        scores[f"score_A_{arm.removeprefix('V2_')}"] = frozen[f"score_{arm}"].to_numpy()
        scores[f"pharm_A_{arm.removeprefix('V2_')}"] = frozen[f"pharm_{arm}"].to_numpy()

    arm_records: dict[str, Any] = {}
    for plan in _fit_plans():
        evidence_config = "sgv1_v0" if plan.label == "V0" else "sgv1_v1"
        mask = EvidenceMask.from_key(evidence_config)
        embedding = _embedding_for(plan, everything)
        verifier = RepresentationVerifier(
            verifier_id=f"sgv1_repr_{plan.label.lower()}",
            evidence_config=evidence_config,
            model="logistic",
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=pilot.FIT_SEED,
            provenance=None if plan.label == "V0" else provenance,
            embedding=embedding,
        )

        fit_inputs, cal_inputs, dev_inputs = (
            pilot._inputs_for(
                plan.label, list(frame["candidate_id"].astype(str)), bundles, {}, mask
            )
            for frame in (fit_rows, cal_rows, dev_rows)
        )
        verifier.fit(fit_inputs, list(fit_rows["is_harmful"].astype(bool)))
        cal_scores = verifier.score(cal_inputs).scores
        dev_scores = verifier.score(dev_inputs).scores

        calibrator = build_calibrator(pilot.CALIBRATION_METHOD)
        calibrator.fit(cal_scores, (~cal_rows["is_harmful"].to_numpy(dtype=bool)).astype(float))
        scores[f"score_{plan.label}"] = dev_scores
        scores[f"pharm_{plan.label}"] = 1.0 - calibrator.transform(dev_scores)

        arm_records[plan.label] = {
            "rung": plan.rung or "reference",
            "crop_scale": plan.scale,
            "seed": plan.seed,
            "arm": plan.arm,
            "evidence_config": evidence_config,
            "feature_dimension": len(verifier.feature_names),
            "feature_names_sha256": _digest(*verifier.feature_names),
            "image_block_columns": EMBED_DIM + 1 if plan.rung else 0,
            "model": "logistic_regression",
            "model_parameters": verifier.parameter_count,
            "trainable_parameters": verifier.parameter_count,
            "calibrator_identity": calibrator.identity(),
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "random_state": pilot.FIT_SEED,
            "calibration_method": pilot.CALIBRATION_METHOD,
            "fit_rows": len(fit_rows),
            "calibration_rows": len(cal_rows),
            "evaluation_rows": len(dev_rows),
        }
        print(f"  fitted {plan.label}: {len(verifier.feature_names)} features")

    # Fairness proof, not fairness claim: this run's V1 must be the Session-2 V1 exactly.
    # If it does not, either the baseline moved or this stage's fitting path differs from
    # the pilot's, and every rung-versus-rung number below would be uninterpretable.
    drift = float(np.max(np.abs(scores["score_V1"].to_numpy() - frozen["score_V1"].to_numpy())))
    v0_drift = float(np.max(np.abs(scores["score_V0"].to_numpy() - frozen["score_V0"].to_numpy())))
    if drift > 1e-12:
        raise RepresentationError(f"re-fitted V1 diverges from the Session-2 V1 by {drift:.3e}")
    if v0_drift > 1e-12:
        raise RepresentationError(f"re-fitted V0 diverges from the Session-2 V0 by {v0_drift:.3e}")
    masked_drift = {
        rung: float(
            np.max(
                np.abs(
                    scores[f"score_{rung}@{BASELINE_SCALE}#{PRIMARY_SEED}_masked"].to_numpy()
                    - scores["score_V1"].to_numpy()
                )
            )
        )
        for rung in ("B", "C")
    }

    _write_parquet_once(SCORES, scores)
    record = {
        "schema_version": "sgv1-representation-fit-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "stage": "DEVELOPMENT",
        "held_out_axis": "documents (TRAIN / CALIBRATION / DEVELOPMENT roles)",
        "held_out_engine": None,
        "limitation": (
            "Every engine appears in every role. Engine-conditional numbers here are "
            "per-engine slices of a document-held-out fit, never held-out-engine results."
        ),
        "arms": arm_records,
        "rung_a_source": "frozen Session-2 scores, merged not refitted",
        "rung_a_parameters": _read_json(BASELINE_RECORD)["model_configuration"]["V2_correct"][
            "model_parameters"
        ],
        "v1_reproduces_session_2": True,
        "v1_max_absolute_drift": drift,
        "v0_reproduces_session_2": True,
        "v0_max_absolute_drift": v0_drift,
        "masked_arm_max_absolute_drift_from_v1": masked_drift,
        "rows_evaluated": len(scores),
        "documents_evaluated": int(scores["document_id"].nunique()),
        "fit_documents": int(fit_rows["document_id"].nunique()),
        "calibration_documents": int(cal_rows["document_id"].nunique()),
        "evaluation_documents": int(dev_rows["document_id"].nunique()),
        "crop_lineage_rows": len(lineage),
        "inputs": {
            _relative(pilot.SCORES_TABLE): file_sha256(pilot.SCORES_TABLE),
            _relative(pilot.LABEL_TABLE): file_sha256(pilot.LABEL_TABLE),
            _relative(pilot.CANDIDATE_TABLE): file_sha256(pilot.CANDIDATE_TABLE),
            _relative(LINEAGE): file_sha256(LINEAGE),
            _relative(BASELINE_RECORD): file_sha256(BASELINE_RECORD),
            _relative(REGISTRY): file_sha256(REGISTRY),
        },
        "artifacts": {_relative(SCORES): file_sha256(SCORES)},
        "confirmatory_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(FIT_RECORD, record)
    print(f"fit: {len(arm_records)} arms, V1 drift {drift:.2e} -> {_relative(SCORES)}")
    return 0


# --------------------------------------------------------- stage 6: analysis

Row = tuple[str, float, float]


def _auc_rows(rows: Sequence[Row]) -> float:
    if not rows:
        return float("nan")
    scores = np.fromiter((r[1] for r in rows), dtype=np.float64, count=len(rows))
    positives = np.fromiter((r[2] for r in rows), dtype=np.float64, count=len(rows))
    if np.unique(positives).size < 2:
        return float("nan")
    return roc_auc(scores, positives)


def _rows_for(frame: pd.DataFrame, column: str, positive: str) -> list[Row]:
    return list(
        zip(
            frame["document_id"].astype(str),
            frame[column].astype(float),
            frame[positive].astype(float),
            strict=True,
        )
    )


def _paired(
    frame: pd.DataFrame, left: str, right: str, positive: str = "beneficial"
) -> dict[str, float]:
    """Δ AUC between two arms on identical documents, resampled by document."""
    result = paired_cluster_bootstrap(
        _rows_for(frame, f"score_{left}", positive),
        _rows_for(frame, f"score_{right}", positive),
        cluster_of=lambda r: r[0],
        statistic=_auc_rows,
        n_resamples=pilot.BOOTSTRAP_RESAMPLES,
        seed=pilot.BOOTSTRAP_SEED,
    )
    return {
        "delta": result.estimate,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "n_documents": result.n_clusters,
    }


def _auc(frame: pd.DataFrame, column: str, positive: str = "beneficial") -> float:
    return pilot._auc_on(frame, column, positive)


def _ap(frame: pd.DataFrame, column: str, positive: str = "beneficial") -> float:
    return pilot._ap_on(frame, column, positive)


def _arm_labels(record: dict[str, Any]) -> list[str]:
    return [*record["arms"], "A_correct", "A_shuffled", "A_masked"]


def _rung_of(label: str) -> str:
    if label in {"V0", "V1"}:
        return "reference"
    return label[0]


def _heterogeneity(deltas: dict[str, float]) -> dict[str, Any]:
    values = [v for v in deltas.values() if np.isfinite(v)]
    if not values:
        return {"classification": "inconclusive", "engines": deltas}
    positive = sum(1 for v in values if v > 0)
    if positive == len(values):
        classification = "consistent positive"
    elif positive == 0:
        classification = "negative"
    elif min(values) < -ENGINE_REGRESSION_MAX:
        classification = "engine-dependent"
    else:
        classification = "mixed"
    return {
        "classification": classification,
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "engines_positive": positive,
        "engines_total": len(values),
        "sign_agreement": positive == len(values) or positive == 0,
        "engines": deltas,
    }


def _encoder_diagnostics() -> dict[str, Any]:
    """Each encoder's image-only held-out AUC, plus what it cost to get it."""
    out: dict[str, Any] = {}
    if not ENCODER_DIR.is_dir():
        return out
    for path in sorted(ENCODER_DIR.glob("*.json")):
        record = _read_json(path)
        out[str(record["encoder"])] = {
            "image_only_inner_dev_roc_auc": record["best_inner_auc"],
            "best_epoch": record["best_epoch"],
            "epochs_run": record["epochs_run"],
            "inner_documents": record["inner_documents"],
            "inner_rows": record["inner_rows"],
            "trainable_parameters": record["trainable_parameters"],
            "runtime_seconds": record["runtime_seconds"],
            "checkpoint_sha256": record["checkpoint_sha256"],
        }
    return out


def run_analyze() -> int:
    started = time.monotonic()
    _assert_registered()
    _assert_baseline_intact()
    record = _read_json(FIT_RECORD)
    if file_sha256(SCORES) != record["artifacts"][_relative(SCORES)]:
        raise RepresentationError("the representation scores moved since the fit record")

    scores = pd.read_parquet(SCORES)
    frame_a = pd.read_parquet(pilot.FRAME_DIR / "frame_a.parquet")
    pairs = pd.read_parquet(pilot.FRAME_DIR / "matched_pairs.parquet")
    ids = set(frame_a["candidate_id"].astype(str))
    in_frame_a = scores[scores["candidate_id"].astype(str).isin(ids)].reset_index(drop=True)
    core = in_frame_a[~in_frame_a["region_is_whitespace_only"]].reset_index(drop=True)
    is_overcorrection = (scores["outcome"] == "overcorrection").to_numpy()
    overcorrection = scores[is_overcorrection]
    beneficial = scores[scores["beneficial"]]
    harmful = scores["is_harmful"].to_numpy(dtype=np.float64)
    engines = sorted(scores["engine_id"].astype(str).unique())
    labels = _arm_labels(record)

    by_arm: dict[str, Any] = {}
    for label in labels:
        column = f"score_{label}"
        probabilities = scores[f"pharm_{label}"].to_numpy(dtype=np.float64)
        lookup = dict(zip(scores["candidate_id"].astype(str), scores[column], strict=True))
        metrics: dict[str, Any] = {
            "rung": _rung_of(label),
            "core_roc_auc": _auc(core, column),
            "core_pr_auc": _ap(core, column),
            "all_roc_auc": _auc(in_frame_a, column),
            "all_pr_auc": _ap(in_frame_a, column),
            "matched_pair_ranking": pilot._pair_ranking(lookup, pairs),
            "brier": brier_score(probabilities, harmful),
            "ece_equal_mass": expected_calibration_error(
                probabilities, harmful, pilot.ECE_BINS, "equal_mass"
            )[0],
            "mean_pharm_overcorrection": float(probabilities[is_overcorrection].mean()),
            "overcorrection_score_gap": float(
                beneficial[column].mean() - overcorrection[column].mean()
            ),
            "by_engine": {},
        }
        for engine in engines:
            mask = scores["engine_id"].astype(str) == engine
            engine_core = core[core["engine_id"].astype(str) == engine]
            engine_pairs = pairs[pairs["engine_id"].astype(str) == engine]
            metrics["by_engine"][engine] = {
                "core_rows": len(engine_core),
                "core_roc_auc": _auc(engine_core, column),
                "all_roc_auc": _auc(
                    in_frame_a[in_frame_a["engine_id"].astype(str) == engine], column
                ),
                "brier": brier_score(
                    probabilities[mask.to_numpy()],
                    scores.loc[mask, "is_harmful"].to_numpy(dtype=np.float64),
                ),
                "matched_pairs": len(engine_pairs),
                "matched_pair_ranking": pilot._pair_ranking(lookup, engine_pairs),
            }
        by_arm[label] = metrics

    correct_arms = [label for label in labels if label.endswith("_correct")]
    contrasts: dict[str, Any] = {}
    for label in correct_arms:
        stem = label.removesuffix("_correct")
        shuffled, wrong_local = f"{stem}_shuffled", f"{stem}_wrong_local"
        entry: dict[str, Any] = {
            "vs_V1_core": _paired(core, label, "V1"),
            "vs_V1_all": _paired(in_frame_a, label, "V1"),
            "brier_delta_vs_V1": by_arm[label]["brier"] - by_arm["V1"]["brier"],
            "pair_delta_vs_V1": by_arm[label]["matched_pair_ranking"]
            - by_arm["V1"]["matched_pair_ranking"],
        }
        if shuffled in by_arm:
            entry["locality_core"] = _paired(core, label, shuffled)
            entry["locality_brier"] = by_arm[label]["brier"] - by_arm[shuffled]["brier"]
            entry["locality_pair"] = (
                by_arm[label]["matched_pair_ranking"] - by_arm[shuffled]["matched_pair_ranking"]
            )
        if wrong_local in by_arm:
            entry["strict_locality_core"] = _paired(core, label, wrong_local)
            entry["strict_locality_pair"] = (
                by_arm[label]["matched_pair_ranking"] - by_arm[wrong_local]["matched_pair_ranking"]
            )
        rung = _rung_of(label)
        prefix = "A" if rung == "A" else f"{rung}@{BASELINE_SCALE}#{PRIMARY_SEED}"
        for control in ("masked", "randproj"):
            name = f"{prefix}_{control}"
            if name in by_arm:
                entry[f"vs_{control}_core"] = (
                    by_arm[label]["core_roc_auc"] - by_arm[name]["core_roc_auc"]
                )
        entry["engine_effect_vs_V1"] = _heterogeneity(
            {
                e: by_arm[label]["by_engine"][e]["core_roc_auc"]
                - by_arm["V1"]["by_engine"][e]["core_roc_auc"]
                for e in engines
            }
        )
        if shuffled in by_arm:
            entry["engine_locality"] = _heterogeneity(
                {
                    e: by_arm[label]["by_engine"][e]["core_roc_auc"]
                    - by_arm[shuffled]["by_engine"][e]["core_roc_auc"]
                    for e in engines
                }
            )
        contrasts[label] = entry
        print(f"  {label}: core Δ vs V1 {entry['vs_V1_core']['delta']:+.5f}")

    payload = {
        "schema_version": "sgv1-representation-metrics-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "primary_frame": "frame_a_core",
        "primary_endpoint": "core Frame-A ROC AUC",
        "central_contrast": "correct minus shuffled",
        "support": {
            "frame_a_rows": len(in_frame_a),
            "frame_a_documents": int(in_frame_a["document_id"].nunique()),
            "core_rows": len(core),
            "core_beneficial": int(core["beneficial"].sum()),
            "core_harmful": int(core["is_harmful"].sum()),
            "core_documents": int(core["document_id"].nunique()),
            "matched_pairs": len(pairs),
            "matched_pair_documents": int(pairs["document_id"].nunique()),
            "overcorrection_rows": len(overcorrection),
            "pool_rows": len(scores),
            "core_rows_by_engine": {
                e: int((core["engine_id"].astype(str) == e).sum()) for e in engines
            },
        },
        "bootstrap": {
            "cluster": "document_id",
            "n_resamples": pilot.BOOTSTRAP_RESAMPLES,
            "seed": pilot.BOOTSTRAP_SEED,
            "paired": True,
        },
        "multiplicity": {
            "control_applied": None,
            "note": "No multiplicity control is applied at development stage. The family "
            f"is {len(labels)} arms x {len(engines)} engines plus {len(contrasts)} "
            "contrast blocks; every interval below is marginal.",
        },
        "by_arm": by_arm,
        "contrasts": contrasts,
        # The single cleanest diagnostic for "does the crop predict harm at all?". Each
        # encoder's held-out inner-dev ROC AUC is measured from the IMAGE ALONE, on TRAIN
        # documents the encoder did not fit. Near 0.5 means the image channel carries
        # nothing the encoder could find, which distinguishes "the fusion washed it out"
        # from "there was never a signal to fuse".
        "encoder_image_only_inner_dev_auc": _encoder_diagnostics(),
        "confirmatory_accessed": False,
        "inputs": {
            _relative(SCORES): file_sha256(SCORES),
            _relative(FIT_RECORD): file_sha256(FIT_RECORD),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(METRICS, payload)
    _write_comparison(payload, record)
    print(f"analyzed: {len(labels)} arms -> {_relative(METRICS)}")
    return 0


def _write_comparison(payload: dict[str, Any], record: dict[str, Any]) -> None:
    """The canonical §25 comparison table, one row per arm."""
    engines = sorted(next(iter(payload["by_arm"].values()))["by_engine"])
    rows: list[dict[str, Any]] = []
    for label, metrics in payload["by_arm"].items():
        arm_record = record["arms"].get(label, {})
        contrast = payload["contrasts"].get(label, {})
        row: dict[str, Any] = {
            "arm": label,
            "rung": metrics["rung"],
            "crop_scale": arm_record.get("crop_scale"),
            "seed": arm_record.get("seed"),
            "core_AUC": metrics["core_roc_auc"],
            "core_PR_AUC": metrics["core_pr_auc"],
            "all_AUC": metrics["all_roc_auc"],
            "pair_accuracy": metrics["matched_pair_ranking"],
            "Brier": metrics["brier"],
            "overcorrection_mean_pharm": metrics["mean_pharm_overcorrection"],
            "correct_minus_V1_core": contrast.get("vs_V1_core", {}).get("delta"),
            "correct_minus_V1_ci_lower": contrast.get("vs_V1_core", {}).get("ci_lower"),
            "correct_minus_V1_ci_upper": contrast.get("vs_V1_core", {}).get("ci_upper"),
            "correct_minus_shuffled_core": contrast.get("locality_core", {}).get("delta"),
            "correct_minus_wrong_local_core": contrast.get("strict_locality_core", {}).get("delta"),
            "correct_minus_masked_core": contrast.get("vs_masked_core"),
            "correct_minus_randproj_core": contrast.get("vs_randproj_core"),
            "engine_effect_class": contrast.get("engine_effect_vs_V1", {}).get("classification"),
            "parameter_count": arm_record.get("model_parameters"),
            "trainable_parameters": arm_record.get("trainable_parameters"),
            "feature_dimension": arm_record.get("feature_dimension"),
        }
        for engine in engines:
            row[f"core_AUC_{engine}"] = metrics["by_engine"][engine]["core_roc_auc"]
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["rung", "arm"]).reset_index(drop=True)
    if COMPARISON.exists():
        raise RepresentationError(f"{_relative(COMPARISON)} already exists; refusing to overwrite")
    COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(COMPARISON, index=False)


# ---------------------------------------------------------- stage 7: figures


def _short(label: str) -> str:
    """Compact axis label. Labels are ``rung@scale#seed_arm``; both the scale
    (``local_medium``) and the arm (``wrong_local``) contain underscores, so the split has
    to follow the separators that are actually unique."""
    if "@" not in label:
        return label
    rung, rest = label.split("@", 1)
    scale, seed_arm = rest.split("#", 1)
    seed, arm = seed_arm.split("_", 1)
    return f"{rung}/{scale.removeprefix('local_')}#{seed[-4:]}/{arm}"


def run_figures() -> int:
    started = time.monotonic()
    metrics = _read_json(METRICS)
    record = _read_json(FIT_RECORD)
    if file_sha256(SCORES) != record["artifacts"][_relative(SCORES)]:
        raise RepresentationError("the representation scores moved since the fit record")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    by_arm = metrics["by_arm"]
    contrasts = metrics["contrasts"]
    correct = sorted(contrasts)
    engines = sorted(next(iter(by_arm.values()))["by_engine"])
    note = "SGV1 DEVELOPMENT -- not a confirmatory result"
    written: list[Path] = []

    def finish(fig: Any, path: Path, title: str) -> None:
        fig.suptitle(f"{title}\n{note}", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    # 1 -- ladder overview
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = [a for a in by_arm if a in {"V0", "V1"}] + correct
    ax.barh([_short(a) for a in order], [by_arm[a]["core_roc_auc"] for a in order], color="#3b6ea5")
    ax.axvline(by_arm["V1"]["core_roc_auc"], color="#c0392b", ls="--", lw=1, label="V1")
    ax.set_xlabel("core Frame-A ROC AUC")
    ax.legend(fontsize=7)
    finish(fig, FIGURE_DIR / "fig01_ladder_overview.png", "Figure 1 -- representation ladder")

    # 2 -- AUC by representation and ablation
    fig, ax = plt.subplots(figsize=(11, 5))
    arms_present = list(IMAGE_ARMS)
    width = 0.16
    positions = np.arange(len(correct))
    for offset, arm in enumerate(arms_present):
        values = []
        for label in correct:
            name = label.removesuffix("_correct") + f"_{arm}"
            values.append(by_arm.get(name, {}).get("core_roc_auc", np.nan))
        ax.bar(positions + offset * width, values, width, label=arm)
    ax.axhline(by_arm["V1"]["core_roc_auc"], color="#c0392b", ls="--", lw=1, label="V1")
    ax.set_xticks(positions + 2 * width)
    ax.set_xticklabels([_short(a) for a in correct], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("core Frame-A ROC AUC")
    ax.set_ylim(min(by_arm["V1"]["core_roc_auc"] - 0.05, 0.7), None)
    ax.legend(fontsize=7, ncol=6)
    finish(fig, FIGURE_DIR / "fig02_auc_by_ablation.png", "Figure 2 -- core AUC by arm")

    # 3 -- locality
    fig, ax = plt.subplots(figsize=(9, 4.5))
    locality = [contrasts[a].get("locality_core", {}).get("delta", np.nan) for a in correct]
    lower = [contrasts[a].get("locality_core", {}).get("ci_lower", np.nan) for a in correct]
    upper = [contrasts[a].get("locality_core", {}).get("ci_upper", np.nan) for a in correct]
    errors = np.abs(np.vstack([np.array(locality) - lower, upper - np.array(locality)]))
    ax.errorbar(locality, np.arange(len(correct)), xerr=errors, fmt="o", color="#2e7d5b", capsize=3)
    ax.axvline(0, color="#666", lw=1)
    ax.axvline(LOCALITY_MIN_AUC, color="#c0392b", ls=":", lw=1, label=f"gate {LOCALITY_MIN_AUC}")
    ax.set_yticks(np.arange(len(correct)))
    ax.set_yticklabels([_short(a) for a in correct], fontsize=7)
    ax.set_xlabel("core ROC AUC:  correct - shuffled")
    ax.legend(fontsize=7)
    finish(fig, FIGURE_DIR / "fig03_locality.png", "Figure 3 -- locality effect")

    # 4 -- matched pairs
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(
        [_short(a) for a in order],
        [by_arm[a]["matched_pair_ranking"] for a in order],
        color="#7d5ba6",
    )
    ax.axvline(by_arm["V1"]["matched_pair_ranking"], color="#c0392b", ls="--", lw=1, label="V1")
    ax.set_xlabel("P(q+ > q-) on 794 within-site pairs")
    ax.set_xlim(0.8, None)
    ax.legend(fontsize=7)
    finish(fig, FIGURE_DIR / "fig04_matched_pairs.png", "Figure 4 -- matched-pair ranking")

    # 5 -- calibration
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].barh([_short(a) for a in order], [by_arm[a]["brier"] for a in order], color="#b5651d")
    axes[0].axvline(by_arm["V1"]["brier"], color="#c0392b", ls="--", lw=1)
    axes[0].set_xlabel("Brier (lower is better)")
    axes[1].barh(
        [_short(a) for a in order], [by_arm[a]["ece_equal_mass"] for a in order], color="#b5651d"
    )
    axes[1].axvline(by_arm["V1"]["ece_equal_mass"], color="#c0392b", ls="--", lw=1)
    axes[1].set_xlabel("ECE, equal-mass, 15 bins")
    finish(fig, FIGURE_DIR / "fig05_calibration.png", "Figure 5 -- calibration")

    # 6 -- overcorrection
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(
        [_short(a) for a in order],
        [by_arm[a]["mean_pharm_overcorrection"] for a in order],
        color="#a33",
    )
    ax.axvline(by_arm["V1"]["mean_pharm_overcorrection"], color="#333", ls="--", lw=1, label="V1")
    ax.set_xlabel("mean calibrated harm probability on clean-anchor overcorrections")
    ax.legend(fontsize=7)
    finish(fig, FIGURE_DIR / "fig06_overcorrection.png", "Figure 6 -- overcorrection behaviour")

    # 7 and 8 -- per engine
    for number, key, title in (
        (7, "engine_effect_vs_V1", "Figure 7 -- per-engine correct-crop effect vs V1"),
        (8, "engine_locality", "Figure 8 -- per-engine locality effect"),
    ):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        positions = np.arange(len(correct))
        for offset, engine in enumerate(engines):
            values = [
                contrasts[a].get(key, {}).get("engines", {}).get(engine, np.nan) for a in correct
            ]
            ax.bar(positions + offset * 0.2, values, 0.2, label=engine)
        ax.axhline(0, color="#666", lw=1)
        ax.set_xticks(positions + 0.3)
        ax.set_xticklabels([_short(a) for a in correct], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("core ROC AUC delta")
        ax.legend(fontsize=7, ncol=4)
        finish(fig, FIGURE_DIR / f"fig0{number}_{key}.png", title)

    # 9 -- capacity versus effect
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in correct:
        params = record["arms"].get(label, {}).get("model_parameters")
        if params is None:
            params = record.get("rung_a_parameters", 56)
        ax.scatter(params, contrasts[label]["vs_V1_core"]["delta"], s=45)
        ax.annotate(_short(label), (params, contrasts[label]["vs_V1_core"]["delta"]), fontsize=6)
    ax.axhline(0, color="#666", lw=1)
    ax.set_xlabel("logistic-regression parameters")
    ax.set_ylabel("core ROC AUC delta vs V1")
    finish(fig, FIGURE_DIR / "fig09_capacity_vs_effect.png", "Figure 9 -- capacity versus effect")

    manifest = {
        "schema_version": "sgv1-representation-figures-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "annotation": note,
        "synthetic": False,
        "confirmatory_accessed": False,
        "derived_from": {
            _relative(METRICS): file_sha256(METRICS),
            _relative(SCORES): file_sha256(SCORES),
            _relative(FIT_RECORD): file_sha256(FIT_RECORD),
        },
        "figures": {_relative(p): file_sha256(p) for p in written},
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(FIGURE_MANIFEST, manifest)
    print(f"figures: {len(written)} -> {_relative(FIGURE_DIR)}")
    return 0


# --------------------------------------------------------- stage 8: decision


def _confirmatory_document_ids() -> set[str]:
    manifest = pilot.load_role_manifest(pilot.ROLE_MANIFEST)
    return {
        str(document_id)
        for document_id, role in manifest["role_of"].items()
        if str(role) == "CONFIRMATORY"
    }


def _confirmatory_documents_absent() -> bool:
    """Intersect the locked reserve against every table this stage produced."""
    reserved = _confirmatory_document_ids()
    if not reserved:
        raise RepresentationError("the role manifest names no confirmatory reserve")
    for path in (SCORES, LINEAGE):
        if not path.is_file():
            continue
        documents = set(pd.read_parquet(path, columns=["document_id"])["document_id"].astype(str))
        if documents & reserved:
            raise RepresentationError(
                f"{len(documents & reserved)} confirmatory documents reached {_relative(path)}"
            )
    return True


def _decide_status(gate_1_passed: bool, candidates: dict[str, Any]) -> tuple[str, str | None, str]:
    """The registered selection rule, as a pure function of the gate results.

    Separated from artifact IO so the four branches can be exercised directly: this is the
    single line the whole phase reduces to, and it should not be reachable only by
    producing a full set of artifacts first.
    """
    passing = [
        lab
        for lab, entry in candidates.items()
        if entry["gate_2_passed"] and entry["gate_4_passed"] and entry["eligible_for_selection"]
    ]
    ranked = sorted(
        passing,
        key=lambda lab: candidates[lab]["gate_3_incremental"]["core_auc_delta_vs_V1"],
        reverse=True,
    )
    any_locality = [
        lab
        for lab, entry in candidates.items()
        if entry["gate_2_passed"] and entry["eligible_for_selection"]
    ]

    if not gate_1_passed:
        return (
            "REPRESENTATION_INCONCLUSIVE",
            None,
            "Gate 1 (validity) failed; no representation comparison is readable.",
        )
    if ranked:
        best = ranked[0]
        return (
            "REPRESENTATION_READY",
            best,
            f"{best} shows a locality effect of {candidates[best]['locality_core']:+.5f} core "
            f"ROC AUC above the registered threshold {LOCALITY_MIN_AUC}, beats its "
            "same-document wrong-crop control, and no engine regresses beyond the "
            "registered bound.",
        )
    if any_locality:
        return (
            "REPRESENTATION_PARTIALLY_READY",
            None,
            f"{len(any_locality)} representation(s) clear the locality gate but fail engine "
            "robustness. The image signal is not uniformly beneficial across engines.",
        )
    strongest = max(candidates, key=lambda lab: candidates[lab].get("locality_core") or -np.inf)
    return (
        "REPRESENTATION_NOT_READY",
        None,
        "No tested representation shows a locality effect above the registered threshold "
        f"{LOCALITY_MIN_AUC}. The strongest was {strongest} at "
        f"{candidates[strongest].get('locality_core')}. Beating V1 without beating an "
        "irrelevant crop is not evidence of source-local information use.",
    )


def _gate_4_power(
    by_arm: dict[str, Any], contrasts: dict[str, Any], selected: str | None
) -> dict[str, Any]:
    """Can the engine-robustness gate actually tell these representations apart?

    Post-hoc, and labelled as such. Gate 4 is evaluated on per-engine core slices of 107
    to 673 rows. If the same engine's estimate swings further across otherwise-identical
    configurations than the gate's own bound, the gate is measuring sampling noise, and a
    unique winner it produces is an artifact of which noise draw happened to land inside
    the bound. The criterion applied here: the gate is underpowered for the selected arm
    when its binding engine's margin is smaller than the standard deviation of that
    engine's effect across the whole ladder.
    """
    engines = sorted(by_arm["V1"]["by_engine"])
    spread: dict[str, Any] = {}
    for engine in engines:
        values = [c["engine_effect_vs_V1"]["engines"][engine] for c in contrasts.values()]
        spread[engine] = {
            "core_rows": by_arm["V1"]["by_engine"][engine]["core_rows"],
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
            "sd_across_arms": float(np.std(values)),
            "sd_exceeds_gate_bound": bool(np.std(values) > ENGINE_REGRESSION_MAX),
        }
    out: dict[str, Any] = {
        "note": (
            "POST-HOC power check, not part of the registered rule. Recorded because the "
            "registered rule's Gate 4 is what reduced the ladder to a single winner."
        ),
        "engine_effect_spread_across_the_ladder": spread,
        "engines_whose_sd_exceeds_the_gate_bound": [
            e for e, v in spread.items() if v["sd_exceeds_gate_bound"]
        ],
    }
    if selected is None:
        out["underpowered_for_selected_arm"] = None
        return out
    engine_effects = contrasts[selected]["engine_effect_vs_V1"]["engines"]
    binding = min(engine_effects, key=lambda e: engine_effects[e])
    margin = engine_effects[binding] + ENGINE_REGRESSION_MAX
    out["selected_arm_binding_engine"] = {
        "engine": binding,
        "effect": engine_effects[binding],
        "margin_inside_bound": margin,
        "core_rows": spread[binding]["core_rows"],
        "sd_across_arms": spread[binding]["sd_across_arms"],
    }
    out["underpowered_for_selected_arm"] = bool(margin < spread[binding]["sd_across_arms"])
    return out


def run_decide() -> int:
    started = time.monotonic()
    registry = _assert_registered()
    _assert_baseline_intact()
    metrics = _read_json(METRICS)
    record = _read_json(FIT_RECORD)
    by_arm, contrasts = metrics["by_arm"], metrics["contrasts"]

    gate_1 = {
        "baseline_binding_intact": True,
        "registry_intact": True,
        "v1_reproduces_session_2": record["v1_reproduces_session_2"],
        "v1_max_absolute_drift_under_1e-12": record["v1_max_absolute_drift"] <= 1e-12,
        "masked_arms_reproduce_v1": all(
            v <= 1e-9 for v in record["masked_arm_max_absolute_drift_from_v1"].values()
        ),
        "same_rows_every_arm": record["rows_evaluated"] == len(pd.read_parquet(SCORES)),
        # Re-derived, not read back. Reviewer C's Medium finding against the Session-2 C3
        # gate was that its reserve check only re-read a boolean this pipeline had written
        # itself, so a regression that let a reserve document through while still writing
        # `confirmatory_accessed: false` would pass. This intersects the locked reserve
        # against the document ids actually present in the produced tables.
        "no_confirmatory_document_in_any_output": _confirmatory_documents_absent(),
        "crop_geometry_from_ocr_only": _read_json(CROP_SCALE_REGISTRY)[
            "geometry_never_from_ground_truth"
        ],
    }
    gate_1_passed = all(gate_1.values())

    candidates: dict[str, Any] = {}
    for label, contrast in contrasts.items():
        # The registry says the primary result is the PRIMARY SEED and the other seeds are
        # stability evidence that "may not be selected among". Ranking over all of them
        # would quietly turn a three-seed spread into a three-way max, which is seed
        # shopping with extra steps -- and here it would matter, because the best-scoring
        # arm is a stability seed and the registered primary seed is the most conservative
        # of the three. Non-primary seeds are still measured and reported; they are just
        # not selectable.
        seed = record["arms"].get(label, {}).get("seed")
        eligible = seed is None or seed == PRIMARY_SEED
        locality = contrast.get("locality_core", {}).get("delta")
        strict = contrast.get("strict_locality_core", {}).get("delta")
        engine = contrast.get("engine_effect_vs_V1", {})
        gate_2 = {
            "locality_core_above_threshold": locality is not None
            and np.isfinite(locality)
            and locality > LOCALITY_MIN_AUC,
            "beats_wrong_local": strict is None or (np.isfinite(strict) and strict > 0),
        }
        gate_4 = {
            "no_engine_regression_beyond_threshold": engine.get("min", -np.inf)
            > -ENGINE_REGRESSION_MAX,
            "classification": engine.get("classification"),
        }
        candidates[label] = {
            "seed": seed,
            "eligible_for_selection": eligible,
            "eligibility_note": (
                "primary seed" if eligible else "stability seed -- reported, not selectable"
            ),
            "gate_2_locality": gate_2,
            "gate_2_passed": all(gate_2.values()),
            "gate_3_incremental": {
                "core_auc_delta_vs_V1": contrast["vs_V1_core"]["delta"],
                "core_auc_ci": [
                    contrast["vs_V1_core"]["ci_lower"],
                    contrast["vs_V1_core"]["ci_upper"],
                ],
                "all_auc_delta_vs_V1": contrast["vs_V1_all"]["delta"],
                "brier_delta_vs_V1": contrast["brier_delta_vs_V1"],
                "pair_delta_vs_V1": contrast["pair_delta_vs_V1"],
                "core_pr_auc": by_arm[label]["core_pr_auc"],
            },
            "gate_4_engine_robustness": gate_4,
            "gate_4_passed": gate_4["no_engine_regression_beyond_threshold"],
            "locality_core": locality,
            "strict_locality_core": strict,
        }

    registered_status, registered_selection, registered_reason = _decide_status(
        gate_1_passed, candidates
    )
    power = _gate_4_power(by_arm, contrasts, registered_selection)

    # The registered rule is preserved verbatim. Where its discriminating gate is shown to
    # be underpowered, the REPORTED status is downgraded and the deviation is recorded as
    # a deviation -- a rule may be departed from, but never silently.
    status, selected, reason = registered_status, registered_selection, registered_reason
    deviation: dict[str, Any] | None = None
    if registered_status == "REPRESENTATION_READY" and power.get("underpowered_for_selected_arm"):
        binding = power["selected_arm_binding_engine"]
        status, selected = "REPRESENTATION_PARTIALLY_READY", None
        reason = (
            f"The registered rule returns REPRESENTATION_READY and selects "
            f"{registered_selection}, but that selection rests entirely on Gate 4, which "
            f"is decided here by {binding['engine']} on {binding['core_rows']} core rows "
            f"with a margin of {binding['margin_inside_bound']:+.5f} against a "
            f"{binding['sd_across_arms']:.5f} standard deviation across the ladder. Gate 4 "
            "cannot discriminate at this support, so the arm-level selection is not "
            "supported by the data. The class-level finding is: every one of the nine arms "
            "cleared the locality gate, and the locality effect accounts for essentially "
            "all of the gain over V1."
        )
        deviation = {
            "type": "post_hoc_downgrade",
            "registered_status": registered_status,
            "registered_selection": registered_selection,
            "registered_reason": registered_reason,
            "reported_status": status,
            "why": (
                "Gate 4's bound is smaller than the sampling variability of the estimate it "
                "is applied to. Reporting a unique winner chosen by that gate would present "
                "a noise draw as a selection."
            ),
            "what_is_still_supported": (
                "Locality: all nine arms cleared the registered locality threshold, with "
                "correct minus shuffled accounting for nearly all of correct minus V1, and "
                "correct minus same-page-wrong-crop generally larger still."
            ),
        }
    ranked = sorted(
        (
            lab
            for lab, e in candidates.items()
            if e["gate_2_passed"] and e["gate_4_passed"] and e["eligible_for_selection"]
        ),
        key=lambda lab: candidates[lab]["gate_3_incremental"]["core_auc_delta_vs_V1"],
        reverse=True,
    )

    decision = {
        "schema_version": "sgv1-representation-decision-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": pilot._git_head(),
        "stage": "DEVELOPMENT",
        "status": status,
        "selected_representation": selected,
        "reason": reason,
        "registered_rule_status": registered_status,
        "registered_rule_selection": registered_selection,
        "deviation_from_registered_rule": deviation,
        "gate_4_power_check": power,
        "gate_1_validity": gate_1,
        "gate_1_passed": gate_1_passed,
        "candidates": candidates,
        "ranking_by_core_auc_delta": ranked,
        "registered_thresholds": {
            "locality_min_core_auc": LOCALITY_MIN_AUC,
            "engine_regression_max": ENGINE_REGRESSION_MAX,
        },
        "registry_sha256": file_sha256(REGISTRY),
        "hypothesis_verdict": None,
        "hypothesis_note": (
            "This stage issues no SGV1-H1 verdict. A development selection is a choice of "
            "what to submit to confirmation, not evidence that the method works."
        ),
        "c3_status": _read_json(pilot.C3_CERTIFICATE)["verdict"],
        "c2_status": "DEFERRED",
        "confirmatory_accessed": False,
        "prohibited_actions_not_taken": registry["prohibited_in_this_stage"],
        "inputs": {
            _relative(METRICS): file_sha256(METRICS),
            _relative(FIT_RECORD): file_sha256(FIT_RECORD),
            _relative(REGISTRY): file_sha256(REGISTRY),
            _relative(BASELINE_RECORD): file_sha256(BASELINE_RECORD),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(DECISION, decision)
    print(f"decision: {status} ({selected or 'no representation selected'})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-baseline", action="store_true")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--crops", action="store_true")
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--decide", action="store_true")
    parser.add_argument("--rung", default=None, help="restrict --encode to one rung")
    parser.add_argument("--scale", default=None, help="restrict --encode to one crop scale")
    args = parser.parse_args()

    if args.bind_baseline:
        return run_bind_baseline()
    if args.register:
        return run_register()
    if args.crops:
        return run_crops()
    if args.encode:
        return run_encode(args.rung, args.scale)
    if args.fit:
        return run_fit()
    if args.analyze:
        return run_analyze()
    if args.figures:
        return run_figures()
    if args.decide:
        return run_decide()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
