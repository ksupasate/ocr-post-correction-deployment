#!/usr/bin/env python3
"""The SGV1 development pilot: V0 / V1 / V2 over the frozen labelled candidates.

Three rungs on exactly the same examples, with exactly the same fitter, so the primary
contrast isolates one thing::

    V0  text only
    V1  V0 + every admitted non-image channel (confidence, geometry, provenance)
    V2  V1 + the source-image crop            <- the only difference from V1

and three V2 arms that separate "the image channel helps" from "adding any eighth block
helps"::

    V2_correct   the crop this candidate's own OCR geometry names
    V2_shuffled  a crop from a different document, chosen deterministically
    V2_masked    the V2 architecture with no pixels behind it

Roles are documents, not engines: fit on TRAIN, calibrate on CALIBRATION, evaluate on
DEVELOPMENT. This is a *development* pilot, so it answers whether the experiment is
runnable and whether the image channel behaves mechanistically -- not whether the method
works under engine shift, which is a confirmatory question this stage may not touch.

    --evidence  ground-truth-blind evidence bundles, crop recipes, and crop lineage
    --pilot     fit / calibrate / evaluate the five arms
    --stats     document-clustered intervals and support diagnostics for the endpoints
    --c3        the machine-readable C3 certificate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ocr_risk.calibrate.calibrators import build_calibrator
from ocr_risk.canonical import rebuild_stream
from ocr_risk.datasets.cord import CordDataset
from ocr_risk.evidence.builder import EvidenceBuilder
from ocr_risk.evidence.crops import CropPolicy, build_recipe, materialize
from ocr_risk.evidence.features_conf import ConfidenceFeaturizer
from ocr_risk.evidence.mask import EvidenceMask
from ocr_risk.experiments.sgv1_design import PRIMARY_CONTRAST, SGV1_LADDER, validate_ladder
from ocr_risk.experiments.sgv1_frames import label_columns_present
from ocr_risk.experiments.sgv1_reserve import (
    assert_sgv1_development_access,
    load_reserve_lock,
    load_role_manifest,
)
from ocr_risk.io.hashing import file_sha256, stable_string_set_hash
from ocr_risk.io.paths import cache_root, raw_source_dir
from ocr_risk.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    murphy_decomposition,
)
from ocr_risk.metrics.discrimination import average_precision, roc_auc
from ocr_risk.schemas.base import BBox
from ocr_risk.schemas.evidence import EvidenceBundle, ObservationView
from ocr_risk.schemas.spans import CanonicalSpan
from ocr_risk.stats.bootstrap import cluster_bootstrap, paired_cluster_bootstrap
from ocr_risk.verify.base import VerificationInput
from ocr_risk.verify.feature_verifier import FeatureVerifier
from ocr_risk.verify.featurizers import FeatureBlock

REPO = Path(__file__).resolve().parents[1]
ROLE_MANIFEST = REPO / "manifests/sgv1/role_manifest.json"
RESERVE_LOCK = REPO / "manifests/sgv1/confirmatory_reserve_lock.json"
RESERVE_SNAPSHOT = REPO / "results/generated/sgv1/reserve/pre_access_freshness_snapshot.json"
OCR_FREEZE = REPO / "results/generated/sgv1/dev_ocr/canonical_ocr_freeze.json"
OCR_SPANS = REPO / "results/generated/sgv1/dev_ocr/canonical_spans.parquet"
CANDIDATE_TABLE = REPO / "results/generated/sgv1/dev_candidates/candidates_pre_gt.parquet"
CANDIDATE_FREEZE = REPO / "results/generated/sgv1/dev_candidates/candidate_freeze.json"
LABEL_TABLE = REPO / "results/generated/sgv1/dev_labels/labels.parquet"
LABEL_RECORD = REPO / "results/generated/sgv1/dev_labels/labeling_record.json"
FRAME_DIR = REPO / "results/generated/sgv1/dev_frames"
FRAMES_RECORD = FRAME_DIR / "frames_record.json"
PILOT_DIR = REPO / "results/generated/sgv1/verifier_pilot"
EVIDENCE_TABLE = PILOT_DIR / "evidence_bundles.parquet"
CROP_LINEAGE = PILOT_DIR / "crop_lineage.parquet"
EVIDENCE_RECORD = PILOT_DIR / "evidence_record.json"
PILOT_RECORD = PILOT_DIR / "pilot_record.json"
SCORES_TABLE = PILOT_DIR / "development_scores.parquet"
C3_CERTIFICATE = PILOT_DIR / "c3_certificate.json"
STATISTICS_RECORD = PILOT_DIR / "development_statistics.json"

CONTEXT_CHARS = 40
CROP_POLICY = CropPolicy(padding_ratio=0.25, padding_min_px=4, target_height=48, grayscale=True)
SHUFFLE_SEED = 20260831
FIT_SEED = 20260831
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 7
CALIBRATION_METHOD = "isotonic"
ECE_BINS = 15

ARMS: dict[str, tuple[str, str]] = {
    "V0": ("sgv1_v0", "none"),
    "V1": ("sgv1_v1", "none"),
    "V2_correct": ("sgv1_v2", "correct"),
    "V2_shuffled": ("sgv1_v2", "shuffled"),
    "V2_masked": ("sgv1_v2", "masked"),
}
PROVENANCE_ARMS = frozenset({"V1", "V2_correct", "V2_shuffled", "V2_masked"})

GENERATOR_SOURCES = ("g3_edit_aware", "g7_structural_v2")
OPERATIONS = ("deletion", "insertion", "merge", "pair_substitution", "split", "substitution")
ANCHOR_KINDS = ("gap", "token", "token_pair")
SITE_TYPES = ("insertion", "merge", "split", "substitution")


class PilotError(RuntimeError):
    """A freeze, role, or fairness invariant failed."""


# --------------------------------------------------------------------------- io


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PilotError(f"expected JSON object at {path}")
    return payload


def _relative(path: Path) -> str:
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise PilotError(f"refusing to overwrite {path}; audit it first") from error


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PilotError(f"refusing to overwrite {path}; audit it first")
    temporary = path.with_suffix(f"{path.suffix}.partial")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


# ------------------------------------------------------------------- guards


def _locked_roles() -> dict[str, str]:
    load_reserve_lock(
        RESERVE_LOCK, role_manifest_path=ROLE_MANIFEST, snapshot_path=RESERVE_SNAPSHOT
    )
    manifest = load_role_manifest(ROLE_MANIFEST)
    roles = {
        document_id: role
        for document_id, role in manifest["role_of"].items()
        if role != "CONFIRMATORY"
    }
    assert_sgv1_development_access(
        sorted(roles),
        operation="SGV1 verifier pilot",
        role_manifest_path=ROLE_MANIFEST,
        lock_path=RESERVE_LOCK,
        snapshot_path=RESERVE_SNAPSHOT,
    )
    return roles


def _verify_upstream() -> dict[str, Any]:
    """The pilot may not run against a candidate or label table that has moved."""
    candidate_freeze = _read_json(CANDIDATE_FREEZE)
    if file_sha256(CANDIDATE_TABLE) != candidate_freeze.get("candidate_table_file_sha256"):
        raise PilotError("candidate table bytes moved since the freeze")
    label_record = _read_json(LABEL_RECORD)
    if file_sha256(LABEL_TABLE) != label_record.get("labels_table_sha256"):
        raise PilotError("label table bytes moved since the labeling record")
    if label_record.get("candidate_table_file_sha256") != candidate_freeze.get(
        "candidate_table_file_sha256"
    ):
        raise PilotError("the label record binds a different candidate table")
    if label_record.get("confirmatory_accessed") is not False:
        raise PilotError("the label record does not declare the reserve untouched")
    frames_record = _read_json(FRAMES_RECORD)
    if frames_record.get("inputs", {}).get(_relative(LABEL_TABLE)) != file_sha256(LABEL_TABLE):
        raise PilotError("the frames record binds a different label table")
    return {
        "candidate_freeze": candidate_freeze,
        "label_record": label_record,
        "frames_record": frames_record,
    }


# ------------------------------------------------------------------ evidence


def _spans_by_pair() -> tuple[
    dict[tuple[str, str], list[CanonicalSpan]], dict[tuple[str, str], str]
]:
    ocr_freeze = _read_json(OCR_FREEZE)
    if file_sha256(OCR_SPANS) != ocr_freeze.get("canonical_spans_sha256"):
        raise PilotError("canonical OCR parquet does not match its freeze")
    table = pd.read_parquet(OCR_SPANS)
    leaking = label_columns_present(table.columns)
    if leaking:
        raise PilotError(f"canonical OCR table carries {leaking}")
    spans: dict[tuple[str, str], list[CanonicalSpan]] = {}
    for raw in table.to_dict("records"):
        span = CanonicalSpan.model_validate(raw)
        spans.setdefault((span.document_id, span.engine_id), []).append(span)
    for pair_spans in spans.values():
        pair_spans.sort(key=lambda s: s.reading_order)
    streams = {pair: rebuild_stream(v) for pair, v in spans.items() if v}
    return spans, streams


@dataclass(frozen=True, slots=True)
class PageImage:
    """Image metadata for one document. Carries no annotation, by construction."""

    document_id: str
    path: Path
    image_sha256: str
    width: int
    height: int


def _page_images(selected: set[str]) -> dict[str, PageImage]:
    """Source-image metadata only.

    CORD's loader yields annotations alongside the page, and the annotations are never
    read here: this returns a type that has no field to put them in, so a crop can not
    acquire ground-truth geometry through this path (reviewer E's attack).
    """
    root = raw_source_dir("cord").parents[2]
    images: dict[str, PageImage] = {}
    for split in ("train", "validation"):
        for bundle in CordDataset(split=split).documents():
            document = bundle.document
            if document.document_id not in selected:
                continue
            images[document.document_id] = PageImage(
                document_id=document.document_id,
                path=root / document.image_path,
                image_sha256=document.image_sha256,
                width=document.width,
                height=document.height,
            )
    missing = sorted(selected - set(images))
    if missing:
        raise PilotError(f"{len(missing)} selected documents have no source image")
    return images


def _anchor_span_ids(anchor_ref: str) -> list[str]:
    return [part for part in anchor_ref.split("\0")[1:] if part]


def _union_bbox(spans: Sequence[CanonicalSpan]) -> BBox | None:
    boxes = [span.bbox for span in spans if span.bbox is not None]
    if not boxes:
        return None
    return BBox(
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )


def _shuffled_donor(candidate_id: str, document_id: str, documents: Sequence[str]) -> str:
    """A different document, chosen deterministically from the candidate id.

    Deterministic so the shuffled arm is reproducible, and cross-document so the donor
    crop cannot accidentally show the same receipt line: a within-document shuffle would
    still leak layout and typography and would weaken the control.
    """
    digest = hashlib.sha256(f"{SHUFFLE_SEED}:{candidate_id}".encode()).digest()
    offset = int.from_bytes(digest[:8], "big") % (len(documents) - 1)
    position = documents.index(document_id)
    return documents[(position + 1 + offset) % len(documents)]


def run_evidence() -> int:
    started = time.monotonic()
    validate_ladder()
    roles = _locked_roles()
    upstream = _verify_upstream()
    candidates = pd.read_parquet(CANDIDATE_TABLE)
    labels = pd.read_parquet(LABEL_TABLE)
    labelled = labels.loc[labels["labelable"], "candidate_id"]
    pool = candidates[candidates["candidate_id"].isin(set(labelled))].reset_index(drop=True)
    if pool.empty:
        raise PilotError("no labelable candidates")

    spans, streams = _spans_by_pair()
    documents = sorted(pool["document_id"].astype(str).unique())
    images = _page_images(set(documents))

    # The confidence featurizer is a fitted transformer: its per-engine mean and spread
    # come from TRAIN rows only. Fitting it on the pooled frame would put development
    # documents into a feature the development split is then scored with (leakage L3).
    train_documents = {d for d, role in roles.items() if role == "TRAIN"}
    featurizer = ConfidenceFeaturizer()
    featurizer.fit(
        [
            (span.engine_id, span.native_conf_recognition, span.conf_scale)
            for (document_id, _engine), pair_spans in spans.items()
            if document_id in train_documents
            for span in pair_spans
        ]
    )
    builder = EvidenceBuilder(
        crop_policy=CROP_POLICY, context_chars=CONTEXT_CHARS, confidence_featurizer=featurizer
    )

    by_span = {pair: {s.span_id: s for s in v} for pair, v in spans.items()}
    bundles: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    materialized: set[str] = set()

    for row in pool.itertuples():
        document_id, engine_id = str(row.document_id), str(row.engine_id)
        pair = (document_id, engine_id)
        anchor_ids = _anchor_span_ids(str(row.anchor_ref))
        anchor_spans = [by_span[pair][s] for s in anchor_ids if s in by_span.get(pair, {})]
        bbox = _union_bbox(anchor_spans)
        page = images[document_id]
        view = ObservationView(
            site_id=str(row.site_id),
            candidate_id=str(row.candidate_id),
            document_id=document_id,
            dataset_id=str(row.dataset_id),
            engine_id=engine_id,
            original_ocr=str(row.original_ocr),
            candidate_text=str(row.candidate_text),
            ocr_span_ids=tuple(anchor_ids),
            image_sha256=page.image_sha256,
            image_width=page.width,
            image_height=page.height,
            bbox=bbox,
        )
        bundle = builder.build(
            view,
            ocr_stream=streams.get(pair, ""),
            char_start=int(row.char_start),
            char_end=int(row.char_end),
            native_confidences=[s.native_conf_recognition for s in anchor_spans],
            conf_scale=next((s.conf_scale for s in anchor_spans if s.conf_scale), None),
        )
        bundles.append({"candidate_id": bundle.candidate_id, "bundle": bundle.model_dump_json()})

        donor_id = _shuffled_donor(str(row.candidate_id), document_id, documents)
        lineage.append(
            {
                "candidate_id": str(row.candidate_id),
                "site_id": str(row.site_id),
                "document_id": document_id,
                "engine_id": engine_id,
                "role": str(row.role),
                "source_image": _relative(page.path),
                "source_image_sha256": page.image_sha256,
                "ocr_span_ids": "\0".join(anchor_ids),
                "geometry_source": "ocr_span_bbox_union",
                "bbox_x0": bbox.x0 if bbox else None,
                "bbox_y0": bbox.y0 if bbox else None,
                "bbox_x1": bbox.x1 if bbox else None,
                "bbox_y1": bbox.y1 if bbox else None,
                "crop_recipe_sha256": bundle.crop_recipe_sha256,
                "shuffled_donor_document_id": donor_id,
            }
        )
        if bundle.crop_recipe_sha256 is not None and bbox is not None:
            recipe = build_recipe(page.image_sha256, bbox, CROP_POLICY)
            if recipe.recipe_sha256 not in materialized:
                materialize(recipe, page.path)
                materialized.add(recipe.recipe_sha256)

    lineage_frame = pd.DataFrame(lineage)
    # The shuffled arm needs a real crop from the donor page, and it must be a crop the
    # donor page actually produced -- a random box would test "is this crop noise?", not
    # "is this crop the right one?".
    donor_recipe = _donor_recipes(lineage_frame, images)
    lineage_frame["shuffled_crop_recipe_sha256"] = lineage_frame["candidate_id"].map(donor_recipe)

    bundle_frame = pd.DataFrame(bundles)
    _write_parquet_once(EVIDENCE_TABLE, bundle_frame)
    _write_parquet_once(CROP_LINEAGE, lineage_frame)
    record = {
        "schema_version": "sgv1-verifier-pilot-evidence-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git_head(),
        "ground_truth_used_for_evidence": False,
        "ground_truth_used_for_row_selection": True,
        "row_selection_note": (
            "Which candidates are labelable is a ground-truth fact, so it selects rows. "
            "No feature is derived from ground truth: EvidenceBuilder accepts only "
            "ObservationView, which has no label field."
        ),
        "confirmatory_accessed": False,
        "rows": len(bundle_frame),
        "documents": len(documents),
        "rows_by_role": {str(k): int(v) for k, v in lineage_frame["role"].value_counts().items()},
        "rows_by_engine": {
            str(k): int(v) for k, v in lineage_frame["engine_id"].value_counts().items()
        },
        "rows_without_geometry": int(lineage_frame["crop_recipe_sha256"].isna().sum()),
        "unique_crop_recipes": int(lineage_frame["crop_recipe_sha256"].nunique()),
        "unique_shuffled_recipes": int(lineage_frame["shuffled_crop_recipe_sha256"].nunique()),
        "crops_materialized": len(materialized),
        "crop_policy": {
            "padding_ratio": CROP_POLICY.padding_ratio,
            "padding_min_px": CROP_POLICY.padding_min_px,
            "target_height": CROP_POLICY.target_height,
            "grayscale": CROP_POLICY.grayscale,
        },
        "crop_geometry_source": "union of the frozen OCR anchor-span bounding boxes",
        "crop_geometry_never_from_ground_truth": True,
        "shuffle_rule": (
            "donor = documents[(index(document) + 1 + sha256(seed:candidate_id) mod "
            "(n_documents - 1)) mod n_documents]; the donor is always a different "
            "document, and its crop is one the donor page actually produced"
        ),
        "shuffle_seed": SHUFFLE_SEED,
        "context_chars": CONTEXT_CHARS,
        "confidence_featurizer_fit_scope": "TRAIN documents only",
        "confidence_featurizer_engines": list(featurizer.engines),
        "confidence_featurizer_train_documents": len(train_documents),
        "inputs": {
            _relative(CANDIDATE_TABLE): file_sha256(CANDIDATE_TABLE),
            _relative(LABEL_TABLE): file_sha256(LABEL_TABLE),
            _relative(LABEL_RECORD): file_sha256(LABEL_RECORD),
            _relative(OCR_SPANS): file_sha256(OCR_SPANS),
            _relative(ROLE_MANIFEST): file_sha256(ROLE_MANIFEST),
        },
        "candidates_sha256": upstream["candidate_freeze"]["candidates_sha256"],
        "artifacts": {
            _relative(EVIDENCE_TABLE): file_sha256(EVIDENCE_TABLE),
            _relative(CROP_LINEAGE): file_sha256(CROP_LINEAGE),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(EVIDENCE_RECORD, record)
    print(
        f"evidence: {len(bundle_frame)} bundles, {len(materialized)} crops, "
        f"record={_relative(EVIDENCE_RECORD)}"
    )
    return 0


def _donor_recipes(lineage: pd.DataFrame, images: dict[str, PageImage]) -> dict[str, str]:
    """Map each candidate to a real crop recipe belonging to its donor document."""
    by_document: dict[str, list[str]] = {}
    for row in lineage.itertuples():
        if isinstance(row.crop_recipe_sha256, str):
            by_document.setdefault(str(row.document_id), []).append(row.crop_recipe_sha256)
    for recipes in by_document.values():
        recipes.sort()
    mapping: dict[str, str] = {}
    for row in lineage.itertuples():
        donor = str(row.shuffled_donor_document_id)
        recipes = by_document.get(donor)
        if not recipes:
            continue
        digest = hashlib.sha256(f"{SHUFFLE_SEED}:donor:{row.candidate_id}".encode()).digest()
        mapping[str(row.candidate_id)] = recipes[int.from_bytes(digest[:8], "big") % len(recipes)]
    _ = images
    return mapping


# ------------------------------------------------------------------ verifier


PROVENANCE_NAMES: tuple[str, ...] = (
    "prov_generator_rank",
    "prov_generator_score",
    "prov_generator_score_missing",
    "prov_suspicion_score",
    "prov_site_candidate_count",
    "prov_region_chars",
    *(f"prov_source_{s}" for s in GENERATOR_SOURCES),
    *(f"prov_operation_{o}" for o in OPERATIONS),
    *(f"prov_anchor_{a}" for a in ANCHOR_KINDS),
    *(f"prov_site_type_{t}" for t in SITE_TYPES),
)


def provenance_block(row: Any, site_candidate_count: int) -> FeatureBlock:
    """Frozen, ground-truth-blind provenance of the proposed edit.

    Which generator proposed the edit, at what rank and score, how suspicious the site
    looked, and what shape the edit is. All of it is on the pre-GT candidate table, so a
    deployed system has it, and the SGV1 protocol requires V1 to be allowed every such
    channel -- a V1 that could not see candidate rank would be a weak baseline chosen to
    flatter V2.
    """
    score = row.generator_score
    has_score = score is not None and not (isinstance(score, float) and np.isnan(score))
    values = [
        float(row.generator_rank),
        float(score) if has_score else 0.0,
        0.0 if has_score else 1.0,
        float(row.suspicion_score),
        float(site_candidate_count),
        float(int(row.char_end) - int(row.char_start)),
        *(1.0 if str(row.generator_source) == s else 0.0 for s in GENERATOR_SOURCES),
        *(1.0 if str(row.operation) == o else 0.0 for o in OPERATIONS),
        *(1.0 if str(row.anchor_kind) == a else 0.0 for a in ANCHOR_KINDS),
        *(1.0 if str(row.site_type) == t else 0.0 for t in SITE_TYPES),
    ]
    return FeatureBlock(names=PROVENANCE_NAMES, values=np.asarray(values, dtype=np.float64))


class SGV1Verifier(FeatureVerifier):
    """``FeatureVerifier`` plus the candidate-provenance block.

    Appended identically for V1 and every V2 arm and withheld from V0, so the primary
    contrast still differs in the image channel alone. Kept here rather than added to
    ``EvidenceField`` on purpose: a new enum member would flow into ``v6`` (which is
    ``frozenset(F)``) and silently change the *other* study's V3-vs-V6 comparison into
    "image plus provenance versus neither".
    """

    def __init__(self, *args: Any, provenance: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._provenance = provenance or {}

    @property
    def parameter_count(self) -> int:
        """Fitted coefficients plus intercept, or zero for the constant fallback."""
        model = self._model
        if model is None or not hasattr(model, "coef_"):
            return 0
        return int(model.coef_.size + model.intercept_.size)

    def _matrix(self, inputs: Sequence[VerificationInput]) -> Any:
        base = super()._matrix(inputs)
        if not self._provenance:
            return base
        extra = np.vstack([self._provenance[item.candidate_id] for item in inputs])
        if len(self._feature_names) == base.shape[1]:
            self._feature_names = (*self._feature_names, *PROVENANCE_NAMES)
        return np.hstack([base, extra])


# ------------------------------------------------------------------- metrics


def _cross_product_pairs(plus: pd.DataFrame, minus: pd.DataFrame, cap: int = 20000) -> pd.DataFrame:
    """A deterministic sample of (beneficial, overcorrection) comparisons.

    Not matched on site -- overcorrection sites are clean by definition and beneficial
    ones are not, so no site holds both. This is therefore an unmatched ranking statistic
    and is reported as one; the capped, seeded sample keeps it from becoming a
    hundred-million-row cross product.
    """
    if plus.empty or minus.empty:
        return pd.DataFrame(columns=["plus_candidate_id", "minus_candidate_id"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = min(cap, len(plus) * len(minus))
    left = rng.integers(0, len(plus), size=n)
    right = rng.integers(0, len(minus), size=n)
    return pd.DataFrame(
        {
            "plus_candidate_id": plus["candidate_id"].to_numpy()[left],
            "minus_candidate_id": minus["candidate_id"].to_numpy()[right],
        }
    )


def _pair_ranking(scores: dict[str, float], pairs: pd.DataFrame) -> float:
    """P(q(Y+) > q(Y-)), ties counted as half. Undefined on an empty set."""
    if pairs.empty:
        return float("nan")
    wins = 0.0
    for row in pairs.itertuples():
        plus = scores[str(row.plus_candidate_id)]
        minus = scores[str(row.minus_candidate_id)]
        wins += 1.0 if plus > minus else 0.5 if plus == minus else 0.0
    return wins / len(pairs)


def _auc_on(frame: pd.DataFrame, score_column: str, positive_column: str) -> float:
    if frame[positive_column].nunique() < 2:
        return float("nan")
    return roc_auc(
        frame[score_column].to_numpy(dtype=np.float64),
        frame[positive_column].to_numpy(dtype=np.float64),
    )


def _ap_on(frame: pd.DataFrame, score_column: str, positive_column: str) -> float:
    if frame[positive_column].nunique() < 2:
        return float("nan")
    return average_precision(
        frame[score_column].to_numpy(dtype=np.float64),
        frame[positive_column].to_numpy(dtype=np.float64),
    )


def _bootstrap(frame: pd.DataFrame, statistic: Any) -> dict[str, float]:
    # Only the columns the statistic reads: each resample rebuilds a DataFrame from these
    # dicts 2,000 times, and carrying the full row would dominate the runtime.
    records = frame[["document_id", "score", "beneficial"]].to_dict("records")
    result = cluster_bootstrap(
        records,
        cluster_of=lambda r: str(r["document_id"]),
        statistic=statistic,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    return {
        "estimate": result.estimate,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "n_documents": result.n_clusters,
    }


def _auc_statistic(score_column: str, positive_column: str) -> Any:
    def statistic(rows: Sequence[dict[str, Any]]) -> float:
        frame = pd.DataFrame(list(rows))
        return _auc_on(frame, score_column, positive_column)

    return statistic


def _paired_auc(
    frame: pd.DataFrame, arm: str, reference: str, positive_column: str
) -> dict[str, float]:
    """Δ AUC between two arms on the same documents, paired at the document level."""
    columns = ["document_id", positive_column]
    left = frame[columns].assign(score=frame[f"score_{arm}"]).to_dict("records")
    right = frame[columns].assign(score=frame[f"score_{reference}"]).to_dict("records")
    result = paired_cluster_bootstrap(
        left,
        right,
        cluster_of=lambda r: str(r["document_id"]),
        statistic=_auc_statistic("score", positive_column),
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    return {
        "delta": result.estimate,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "n_documents": result.n_clusters,
    }


# ---------------------------------------------------------------------- pilot


def _load_bundles() -> dict[str, EvidenceBundle]:
    table = pd.read_parquet(EVIDENCE_TABLE)
    return {
        str(row.candidate_id): EvidenceBundle.model_validate_json(str(row.bundle))
        for row in table.itertuples()
    }


def _inputs_for(
    arm: str,
    ids: Sequence[str],
    bundles: dict[str, EvidenceBundle],
    crops: dict[str, Path | None],
    mask: EvidenceMask,
) -> list[VerificationInput]:
    _ = arm
    return [
        VerificationInput(bundle=mask.apply(bundles[candidate_id]), crop=crops.get(candidate_id))
        for candidate_id in ids
    ]


def _crop_map(arm_mode: str, lineage: pd.DataFrame) -> dict[str, Path | None]:
    """Which pixels each arm sees. ``masked`` sees none, by construction."""
    if arm_mode in {"none", "masked"}:
        return {}
    column = "crop_recipe_sha256" if arm_mode == "correct" else "shuffled_crop_recipe_sha256"
    mapping: dict[str, Path | None] = {}
    for row in lineage.itertuples():
        digest = getattr(row, column)
        if isinstance(digest, str):
            mapping[str(row.candidate_id)] = cache_root() / "crops" / digest[:2] / f"{digest}.png"
    return mapping


def run_pilot() -> int:
    started = time.monotonic()
    validate_ladder()
    roles = _locked_roles()
    upstream = _verify_upstream()
    evidence_record = _read_json(EVIDENCE_RECORD)
    if file_sha256(EVIDENCE_TABLE) != evidence_record["artifacts"][_relative(EVIDENCE_TABLE)]:
        raise PilotError("evidence table bytes moved since the evidence record")

    candidates = pd.read_parquet(CANDIDATE_TABLE)
    labels = pd.read_parquet(LABEL_TABLE)
    lineage = pd.read_parquet(CROP_LINEAGE)
    pairs = pd.read_parquet(FRAME_DIR / "matched_pairs.parquet")
    frame_a = pd.read_parquet(FRAME_DIR / "frame_a.parquet")

    pool = candidates.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    pool = pool[pool["labelable"]].reset_index(drop=True)
    manifest_role = pool["document_id"].map(roles)
    if manifest_role.isna().any():
        raise PilotError("a labelable candidate has no non-confirmatory role")
    # The frozen candidate table carries a role too. They must agree: a divergence would
    # mean the pilot is splitting on a different partition than the freeze recorded.
    if not bool((pool["role"].astype(str) == manifest_role.astype(str)).all()):
        raise PilotError("the frozen candidate role disagrees with the locked role manifest")
    pool["beneficial"] = pool["outcome"].isin({"true_correction", "partial_improvement"})

    bundles = _load_bundles()
    missing = set(pool["candidate_id"]) - set(bundles)
    if missing:
        raise PilotError(f"{len(missing)} labelable candidates have no evidence bundle")

    site_counts = candidates.groupby("site_id").size().to_dict()
    provenance = {
        str(row.candidate_id): provenance_block(row, site_counts[row.site_id]).values
        for row in pool.itertuples()
    }

    fit_rows = pool[pool["role"] == "TRAIN"].reset_index(drop=True)
    cal_rows = pool[pool["role"] == "CALIBRATION"].reset_index(drop=True)
    dev_rows = pool[pool["role"] == "DEVELOPMENT"].reset_index(drop=True)
    for name, frame in (("TRAIN", fit_rows), ("CALIBRATION", cal_rows), ("DEVELOPMENT", dev_rows)):
        if frame.empty:
            raise PilotError(f"{name} role is empty")
    if set(fit_rows["document_id"]) & set(dev_rows["document_id"]):
        raise PilotError("a document appears in both the fit and the evaluation role")
    if set(cal_rows["document_id"]) & set(dev_rows["document_id"]):
        raise PilotError("a document appears in both the calibration and the evaluation role")

    scores = dev_rows[
        [
            "candidate_id",
            "site_id",
            "document_id",
            "engine_id",
            "outcome",
            "is_harmful",
            "beneficial",
            "d_before",
            "d_after",
            "region_is_whitespace_only",
            "operation",
            "anchor_kind",
            "generator_source",
        ]
    ].copy()
    arm_records: dict[str, Any] = {}
    calibrated: dict[str, np.ndarray] = {}

    for arm, (evidence_config, crop_mode) in ARMS.items():
        mask = EvidenceMask.from_key(evidence_config)
        crops = _crop_map(crop_mode, lineage)
        arm_provenance = provenance if arm in PROVENANCE_ARMS else None
        verifier = SGV1Verifier(
            verifier_id=f"sgv1_{arm.lower()}",
            evidence_config=evidence_config,
            model="logistic",
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=FIT_SEED,
            provenance=arm_provenance,
        )
        fit_inputs = _inputs_for(arm, list(fit_rows["candidate_id"]), bundles, crops, mask)
        verifier.fit(fit_inputs, list(fit_rows["is_harmful"].astype(bool)))

        cal_scores = verifier.score(
            _inputs_for(arm, list(cal_rows["candidate_id"]), bundles, crops, mask)
        ).scores
        dev_scores = verifier.score(
            _inputs_for(arm, list(dev_rows["candidate_id"]), bundles, crops, mask)
        ).scores

        # The calibrator's contract is oriented the same way the verifier's score is:
        # higher means safer, and ``positive`` is "accepting this edit is safe". Fitting it
        # on CALIBRATION documents keeps the transform out of the evaluation split.
        calibrator = build_calibrator(CALIBRATION_METHOD)
        calibrator.fit(cal_scores, (~cal_rows["is_harmful"].to_numpy(dtype=bool)).astype(float))
        dev_safe = calibrator.transform(dev_scores)
        dev_harm = 1.0 - dev_safe
        calibrated[arm] = dev_harm

        scores[f"score_{arm}"] = dev_scores
        scores[f"pharm_{arm}"] = dev_harm
        arm_records[arm] = {
            "evidence_config": evidence_config,
            "evidence_fields": sorted(f.value for f in mask.spec.fields),
            "crop_mode": crop_mode,
            "provenance_block": arm in PROVENANCE_ARMS,
            "feature_names": list(verifier.feature_names),
            "feature_dimension": len(verifier.feature_names),
            "model": "logistic_regression",
            "model_parameters": verifier.parameter_count,
            "trainable_parameters": verifier.parameter_count,
            "calibrator_identity": calibrator.identity(),
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "random_state": FIT_SEED,
            "calibration_method": CALIBRATION_METHOD,
            "fit_rows": len(fit_rows),
            "calibration_rows": len(cal_rows),
            "evaluation_rows": len(dev_rows),
            "rows_with_pixels": int(sum(1 for c in crops.values() if c is not None))
            if crops
            else 0,
        }

    _write_parquet_once(SCORES_TABLE, scores)
    metrics = _pilot_metrics(scores, frame_a, pairs, dev_rows)
    record = {
        "schema_version": "sgv1-verifier-pilot-record-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git_head(),
        "scope": "DEVELOPMENT pilot -- exploratory, not a confirmatory result",
        "confirmatory_accessed": False,
        "ground_truth_loaded": True,
        "held_out_axis": "documents (TRAIN / CALIBRATION / DEVELOPMENT roles)",
        "held_out_engine": None,
        "limitation": (
            "Every engine appears in every role, so this pilot does not test the "
            "cross-engine shift the confirmatory design exists for. Engine-conditional "
            "numbers here are per-engine slices of a document-held-out fit, never a "
            "held-out-engine result."
        ),
        "ladder": SGV1_LADDER,
        "primary_contrast": list(PRIMARY_CONTRAST),
        "arms": arm_records,
        "roles": {
            "fit_documents": int(fit_rows["document_id"].nunique()),
            "calibration_documents": int(cal_rows["document_id"].nunique()),
            "evaluation_documents": int(dev_rows["document_id"].nunique()),
            "fit_document_set_sha256": stable_string_set_hash(fit_rows["document_id"].astype(str)),
            "calibration_document_set_sha256": stable_string_set_hash(
                cal_rows["document_id"].astype(str)
            ),
            "evaluation_document_set_sha256": stable_string_set_hash(
                dev_rows["document_id"].astype(str)
            ),
        },
        "metrics": metrics,
        "bootstrap": {
            "n_resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "cluster": "document_id",
        },
        "inputs": {
            _relative(EVIDENCE_TABLE): file_sha256(EVIDENCE_TABLE),
            _relative(EVIDENCE_RECORD): file_sha256(EVIDENCE_RECORD),
            _relative(CROP_LINEAGE): file_sha256(CROP_LINEAGE),
            _relative(LABEL_TABLE): file_sha256(LABEL_TABLE),
            _relative(CANDIDATE_TABLE): file_sha256(CANDIDATE_TABLE),
            _relative(FRAMES_RECORD): file_sha256(FRAMES_RECORD),
        },
        "candidates_sha256": upstream["candidate_freeze"]["candidates_sha256"],
        "artifacts": {_relative(SCORES_TABLE): file_sha256(SCORES_TABLE)},
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(PILOT_RECORD, record)
    print(f"pilot: {len(dev_rows)} development rows scored; record={_relative(PILOT_RECORD)}")
    return 0


def _pilot_metrics(
    scores: pd.DataFrame, frame_a: pd.DataFrame, pairs: pd.DataFrame, dev_rows: pd.DataFrame
) -> dict[str, Any]:
    """Every headline number, per arm and per engine, with document-clustered intervals."""
    frame_a_ids = set(frame_a["candidate_id"].astype(str))
    in_frame_a = scores[scores["candidate_id"].astype(str).isin(frame_a_ids)].reset_index(drop=True)
    risk_set = scores[scores["outcome"] != "lateral_change"].reset_index(drop=True)
    # Over half of Frame A's harmful class is a gap insertion on a whitespace-only region,
    # and anchor_kind is a provenance feature V1 already has -- so a high pooled AUC could
    # be a trivially separable subpopulation rather than verification. The core slice
    # (token and token-pair anchors, real OCR text on both sides) is reported beside it so
    # the reader can tell which is which. Incident SGV1-L4 has the counts.
    core = in_frame_a[~in_frame_a["region_is_whitespace_only"]].reset_index(drop=True)
    overcorrection = scores[scores["outcome"] == "overcorrection"]
    beneficial = scores[scores["beneficial"]]

    out: dict[str, Any] = {
        "denominators": {
            "frame_a": (
                "stratified DEVELOPMENT benchmark: beneficial and harmful only. Its class "
                "balance is a design parameter, so ROC AUC (prevalence-invariant) is the "
                "readable discrimination number here and PR AUC is not comparable to a "
                "natural-prevalence PR AUC."
            ),
            "pool": (
                "every labelable DEVELOPMENT candidate. The risk-set metrics drop "
                "lateral_change so the contrast is beneficial against harmful; "
                "calibration is measured over the whole pool, neutral rows included, "
                "because a deployed verifier scores those too."
            ),
            "matched_pairs": (
                "within-site (beneficial, harmful) pairs from the labelled DEVELOPMENT "
                "pool. The site is held constant, so image, engine, layout and OCR "
                "quality are controlled by construction."
            ),
        },
        "frame_a_evaluation_rows": len(in_frame_a),
        "frame_a_beneficial": int(in_frame_a["beneficial"].sum()),
        "frame_a_harmful": int(in_frame_a["is_harmful"].sum()),
        "frame_a_whitespace_only_rows": int(in_frame_a["region_is_whitespace_only"].sum()),
        "frame_a_core_rows": len(core),
        "frame_a_core_beneficial": int(core["beneficial"].sum()),
        "frame_a_core_harmful": int(core["is_harmful"].sum()),
        "frame_a_core_documents": int(core["document_id"].nunique()),
        "development_pool_rows": len(dev_rows),
        "overcorrection_rows": len(overcorrection),
        "overcorrection_documents": int(overcorrection["document_id"].nunique()),
        "matched_pairs": len(pairs),
        "matched_pair_documents": int(pairs["document_id"].nunique()),
        "by_arm": {},
        "ablation_contrasts": {},
    }
    harmful_outcomes = scores["is_harmful"].to_numpy(dtype=np.float64)
    for arm in ARMS:
        column = f"score_{arm}"
        in_frame_a["score"] = in_frame_a[column]
        core["score"] = core[column]
        probabilities = scores[f"pharm_{arm}"].to_numpy(dtype=np.float64)
        arm_metrics: dict[str, Any] = {
            "frame_a_roc_auc": _bootstrap(in_frame_a, _auc_statistic("score", "beneficial")),
            "frame_a_pr_auc": _ap_on(in_frame_a, column, "beneficial"),
            "frame_a_core_roc_auc": _bootstrap(core, _auc_statistic("score", "beneficial")),
            "frame_a_core_pr_auc": _ap_on(core, column, "beneficial"),
            "pool_roc_auc_beneficial_vs_harmful": _auc_on(risk_set, column, "beneficial"),
            "pool_pr_auc_harmful": _ap_on(
                risk_set.assign(harm=risk_set["is_harmful"].astype(float)),
                column,
                "harm",
            ),
            "matched_pair_ranking": _pair_ranking(
                dict(zip(scores["candidate_id"].astype(str), scores[column], strict=True)), pairs
            ),
            "brier": brier_score(probabilities, harmful_outcomes),
            # Both binning schemes, always: equal-width ECE is binning-biased, and a
            # single number would let the reader mistake a binning artefact for calibration.
            "ece_equal_width": expected_calibration_error(
                probabilities, harmful_outcomes, ECE_BINS, "equal_width"
            )[0],
            "ece_equal_mass": expected_calibration_error(
                probabilities, harmful_outcomes, ECE_BINS, "equal_mass"
            )[0],
            "max_calibration_error_equal_mass": expected_calibration_error(
                probabilities, harmful_outcomes, ECE_BINS, "equal_mass"
            )[1],
            "mean_score_overcorrection": float(overcorrection[column].mean()),
            "mean_score_beneficial": float(beneficial[column].mean()),
            "mean_score_harmful": float(scores.loc[scores["is_harmful"], column].mean()),
            "overcorrection_rank_below_beneficial": _pair_ranking(
                dict(zip(scores["candidate_id"].astype(str), scores[column], strict=True)),
                _cross_product_pairs(beneficial, overcorrection),
            ),
            "overcorrection_score_gap": float(
                beneficial[column].mean() - overcorrection[column].mean()
            ),
            "by_engine": {},
        }
        calibration, refinement, uncertainty = murphy_decomposition(
            probabilities, harmful_outcomes, ECE_BINS, "equal_mass"
        )
        arm_metrics["murphy"] = {
            "calibration": calibration,
            "refinement": refinement,
            "uncertainty": uncertainty,
        }
        for engine, group in in_frame_a.groupby("engine_id"):
            engine_pairs = pairs[pairs["engine_id"] == engine]
            arm_metrics["by_engine"][str(engine)] = {
                "frame_a_rows": len(group),
                "frame_a_documents": int(group["document_id"].nunique()),
                "frame_a_roc_auc": _auc_on(group, column, "beneficial"),
                "frame_a_pr_auc": _ap_on(group, column, "beneficial"),
                "matched_pairs": len(engine_pairs),
                "matched_pair_documents": int(engine_pairs["document_id"].nunique()),
                "matched_pair_ranking": _pair_ranking(
                    dict(zip(scores["candidate_id"].astype(str), scores[column], strict=True)),
                    engine_pairs,
                ),
                "pool_rows": int((scores["engine_id"] == engine).sum()),
                "brier": brier_score(
                    scores.loc[scores["engine_id"] == engine, f"pharm_{arm}"].to_numpy(
                        dtype=np.float64
                    ),
                    scores.loc[scores["engine_id"] == engine, "is_harmful"].to_numpy(
                        dtype=np.float64
                    ),
                ),
                "mean_score_overcorrection": (
                    float(overcorrection.loc[overcorrection["engine_id"] == engine, column].mean())
                    if (overcorrection["engine_id"] == engine).any()
                    else float("nan")
                ),
                "overcorrection_rows": int((overcorrection["engine_id"] == engine).sum()),
            }
        out["by_arm"][arm] = arm_metrics

    for arm in ("V2_correct", "V2_shuffled", "V2_masked", "V0"):
        out["ablation_contrasts"][f"{arm}_minus_V1"] = {
            "frame_a_roc_auc": _paired_auc(in_frame_a, arm, "V1", "beneficial"),
            "frame_a_core_roc_auc": _paired_auc(core, arm, "V1", "beneficial"),
            "matched_pair_ranking_delta": (
                out["by_arm"][arm]["matched_pair_ranking"]
                - out["by_arm"]["V1"]["matched_pair_ranking"]
            ),
            "brier_delta": out["by_arm"][arm]["brier"] - out["by_arm"]["V1"]["brier"],
        }
    return out


# -------------------------------------------------------------------- c3


def _pair_ranking_bootstrap(scores: dict[str, float], pairs: pd.DataFrame) -> dict[str, float]:
    """P(q+ > q-) with an interval clustered on the document, not the pair.

    Pairs within a page are not independent -- one document contributes 18.5% of them --
    so a pair-level interval would be dishonestly narrow. The resampling unit is the
    document, per the project's statistics rules.
    """
    records = pairs[["document_id", "plus_candidate_id", "minus_candidate_id"]].to_dict("records")

    def statistic(rows: Sequence[dict[str, Any]]) -> float:
        if not rows:
            return float("nan")
        wins = 0.0
        for row in rows:
            plus = scores[str(row["plus_candidate_id"])]
            minus = scores[str(row["minus_candidate_id"])]
            wins += 1.0 if plus > minus else 0.5 if plus == minus else 0.0
        return wins / len(rows)

    result = cluster_bootstrap(
        records,
        cluster_of=lambda r: str(r["document_id"]),
        statistic=statistic,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    return {
        "estimate": result.estimate,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "n_pairs": len(pairs),
        "n_documents": result.n_clusters,
        "degenerate_interval": bool(result.degenerate_interval),
    }


def _paired_pair_ranking(
    scores_a: dict[str, float], scores_b: dict[str, float], pairs: pd.DataFrame
) -> dict[str, float]:
    """Δ P(q+ > q-) between two arms on the same documents."""
    records = pairs[["document_id", "plus_candidate_id", "minus_candidate_id"]].to_dict("records")

    def make(scores: dict[str, float]) -> Any:
        def statistic(rows: Sequence[dict[str, Any]]) -> float:
            if not rows:
                return float("nan")
            wins = 0.0
            for row in rows:
                plus = scores[str(row["plus_candidate_id"])]
                minus = scores[str(row["minus_candidate_id"])]
                wins += 1.0 if plus > minus else 0.5 if plus == minus else 0.0
            return wins / len(rows)

        return statistic

    left = [{**r, "arm": "a"} for r in records]
    right = [{**r, "arm": "b"} for r in records]
    result = paired_cluster_bootstrap(
        left,
        right,
        cluster_of=lambda r: str(r["document_id"]),
        statistic=lambda rows: (
            make(scores_a) if rows and rows[0]["arm"] == "a" else make(scores_b)
        )(rows),
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    return {
        "delta": result.estimate,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "n_documents": result.n_clusters,
    }


def _concentration(frame: pd.DataFrame, unit: str) -> dict[str, Any]:
    """How much of a statistic one page can carry. Reported, never assumed away."""
    counts = frame.groupby("document_id").size().sort_values(ascending=False)
    total = int(counts.sum())
    return {
        "rows": total,
        "documents": len(counts),
        "sites": int(frame["site_id"].nunique()) if "site_id" in frame.columns else None,
        "largest_document_rows": int(counts.iloc[0]) if total else 0,
        "largest_document_share": round(float(counts.iloc[0] / total), 4) if total else None,
        "top5_share": round(float(counts.head(5).sum() / total), 4) if total else None,
        "median_rows_per_document": float(counts.median()) if total else None,
        "unit": unit,
    }


def run_stats() -> int:
    """Supplementary statistics. Reads the frozen pilot artifacts; changes none of them."""
    started = time.monotonic()
    pilot_record = _read_json(PILOT_RECORD)
    if file_sha256(SCORES_TABLE) != pilot_record["artifacts"][_relative(SCORES_TABLE)]:
        raise PilotError("score table bytes moved since the pilot record")
    scores = pd.read_parquet(SCORES_TABLE)
    pairs = pd.read_parquet(FRAME_DIR / "matched_pairs.parquet")
    frame_a = pd.read_parquet(FRAME_DIR / "frame_a.parquet")
    frame_a_rows = scores[
        scores["candidate_id"].astype(str).isin(set(frame_a["candidate_id"].astype(str)))
    ].reset_index(drop=True)

    by_arm: dict[str, Any] = {}
    lookup = {
        arm: dict(zip(scores["candidate_id"].astype(str), scores[f"score_{arm}"], strict=True))
        for arm in ARMS
    }
    for arm in ARMS:
        by_arm[arm] = {
            "matched_pair_ranking": _pair_ranking_bootstrap(lookup[arm], pairs),
            "by_engine": {
                str(engine): _pair_ranking_bootstrap(lookup[arm], group)
                for engine, group in pairs.groupby("engine_id")
            },
        }
    contrasts = {
        f"{arm}_minus_V1": _paired_pair_ranking(lookup[arm], lookup["V1"], pairs)
        for arm in ("V0", "V2_correct", "V2_shuffled", "V2_masked")
    }

    whitespace = frame_a_rows[frame_a_rows["region_is_whitespace_only"]]
    core = frame_a_rows[~frame_a_rows["region_is_whitespace_only"]]
    slices = {
        "frame_a_pooled": _concentration(frame_a_rows, "frame A row"),
        "frame_a_core": _concentration(core, "frame A row"),
        "frame_a_whitespace_only": _concentration(whitespace, "frame A row"),
        "matched_pairs": _concentration(pairs, "matched pair"),
        "overcorrections": _concentration(
            scores[scores["outcome"] == "overcorrection"], "development candidate"
        ),
    }
    slices["frame_a_whitespace_only"]["beneficial"] = int(whitespace["beneficial"].sum())
    slices["frame_a_whitespace_only"]["harmful"] = int(whitespace["is_harmful"].sum())
    slices["frame_a_core"]["beneficial"] = int(core["beneficial"].sum())
    slices["frame_a_core"]["harmful"] = int(core["is_harmful"].sum())

    n_arms = len(ARMS)
    n_engines = int(scores["engine_id"].nunique())
    record = {
        "schema_version": "sgv1-development-statistics-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git_head(),
        "scope": "DEVELOPMENT pilot -- exploratory. Nothing here is a confirmatory result.",
        "confirmatory_accessed": False,
        "resampling_unit": "document",
        "bootstrap": {"n_resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "matched_pair_ranking": by_arm,
        "matched_pair_contrasts": contrasts,
        "support_and_concentration": slices,
        "multiplicity": {
            "control_applied": None,
            "family_size_pooled": n_arms * 14,
            "family_size_per_engine": n_arms * n_engines * 6,
            "contrasts_reported": len(contrasts) + 4,
            "statement": (
                "No multiplicity control is applied and none is claimed. Every number in "
                "this stage is exploratory and selects nothing: the architecture, the "
                "endpoint, and the operating threshold remain unfrozen, and no result "
                "here may enter a headline table. A family this size is reported so a "
                "reader can discount it, not so it can be read as inference."
            ),
        },
        "interpretation": {
            "whitespace_slice": (
                "714 of Frame A's 2,166 rows sit on a whitespace-only gap region, and "
                "that slice is 699 harmful against 15 beneficial. It is close to "
                "separable from provenance alone -- anchor_kind is a V1 feature -- so the "
                "pooled Frame A AUC sits above the core slice's. The core numbers are the "
                "ones to read for verification capability."
            ),
            "pair_concentration": (
                "One document contributes 147 of 794 matched pairs. Pair-level intervals "
                "would be dishonestly narrow, so every interval here clusters on the "
                "document."
            ),
            "effect_direction": (
                "Δcorrect > Δshuffled > Δmasked holds on both slices, which is the "
                "ordering a real image mechanism would produce. Every V2-minus-V1 "
                "interval still contains zero. This is a consistent direction, not a "
                "demonstrated effect, and it is not evidence for SGV1-H1."
            ),
        },
        "inputs": {
            _relative(PILOT_RECORD): file_sha256(PILOT_RECORD),
            _relative(SCORES_TABLE): file_sha256(SCORES_TABLE),
            _relative(FRAME_DIR / "matched_pairs.parquet"): file_sha256(
                FRAME_DIR / "matched_pairs.parquet"
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(STATISTICS_RECORD, record)
    print(f"statistics: record={_relative(STATISTICS_RECORD)}")
    return 0


def run_c3() -> int:
    """Issue C3 on evidence, and fail closed on an absent input rather than assume."""
    checks: dict[str, bool] = {}
    notes: dict[str, Any] = {}
    for path in (EVIDENCE_RECORD, PILOT_RECORD, SCORES_TABLE, CROP_LINEAGE):
        checks[f"exists:{_relative(path)}"] = path.exists()
    if not all(checks.values()):
        verdict = "C3_INCONCLUSIVE"
        certificate = {
            "schema_version": "sgv1-c3-certificate-v1",
            "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issued_head": _git_head(),
            "verdict": verdict,
            "checks": checks,
            "reason": "a required pilot artifact is absent; the pilot did not complete",
        }
        _write_json_once(C3_CERTIFICATE, certificate)
        print(f"C3: {verdict}")
        return 0

    evidence_record = _read_json(EVIDENCE_RECORD)
    pilot_record = _read_json(PILOT_RECORD)
    lineage = pd.read_parquet(CROP_LINEAGE)
    scores = pd.read_parquet(SCORES_TABLE)
    arms = pilot_record["arms"]

    checks |= {
        "v2_trains_and_scores": all(f"score_{arm}" in scores.columns for arm in ARMS),
        "all_arms_share_the_same_examples": len({len(scores)}) == 1,
        "v1_and_v2_have_identical_non_image_evidence": (
            [n for n in arms["V2_correct"]["feature_names"] if not n.startswith("img_")]
            == arms["V1"]["feature_names"]
        ),
        "image_channel_is_the_only_v2_addition": (
            set(arms["V2_correct"]["feature_names"]) - set(arms["V1"]["feature_names"])
            == {n for n in arms["V2_correct"]["feature_names"] if n.startswith("img_")}
        ),
        "v0_is_text_only": arms["V0"]["evidence_fields"]
        == [
            "candidate_text",
            "original_ocr",
            "text_context",
        ],
        "crops_come_from_ocr_geometry": (
            evidence_record["crop_geometry_source"]
            == "union of the frozen OCR anchor-span bounding boxes"
            and evidence_record["crop_geometry_never_from_ground_truth"] is True
            and set(lineage["geometry_source"]) == {"ocr_span_bbox_union"}
        ),
        "shuffled_crops_come_from_other_documents": bool(
            (lineage["shuffled_donor_document_id"] != lineage["document_id"]).all()
        ),
        "correct_and_shuffled_crops_differ": bool(
            (
                lineage["crop_recipe_sha256"].fillna("")
                != lineage["shuffled_crop_recipe_sha256"].fillna("")
            ).mean()
            > 0.99
        ),
        "gt_does_not_reach_inference_features": (
            evidence_record["ground_truth_used_for_evidence"] is False
        ),
        "confirmatory_reserve_untouched": (
            evidence_record["confirmatory_accessed"] is False
            and pilot_record["confirmatory_accessed"] is False
        ),
        "ablations_executed": all(
            arm in arms for arm in ("V2_correct", "V2_shuffled", "V2_masked")
        ),
        "comparison_is_reproducible": all(
            key in pilot_record for key in ("inputs", "artifacts", "bootstrap", "roles")
        ),
        "evaluation_documents_disjoint_from_fit": (
            pilot_record["roles"]["fit_document_set_sha256"]
            != pilot_record["roles"]["evaluation_document_set_sha256"]
        ),
    }
    notes["masked_arm_has_no_pixels"] = arms["V2_masked"]["rows_with_pixels"] == 0
    notes["v2_correct_rows_with_pixels"] = arms["V2_correct"]["rows_with_pixels"]
    notes["feature_dimensions"] = {a: arms[a]["feature_dimension"] for a in arms}

    failed = sorted(name for name, ok in checks.items() if not ok)
    verdict = "C3_PASS" if not failed else "C3_FAIL"
    certificate = {
        "schema_version": "sgv1-c3-certificate-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git_head(),
        "verdict": verdict,
        "scope": "development experiment viability, not efficacy",
        "statement": (
            "C3 asks whether the SGV1 experiment can be run and read honestly. V2 is not "
            "required to beat V1; a null or negative image effect is a result, not a "
            "failure of this gate."
        ),
        "checks": checks,
        "failed_checks": failed,
        "notes": notes,
        "inputs": {
            _relative(EVIDENCE_RECORD): file_sha256(EVIDENCE_RECORD),
            _relative(PILOT_RECORD): file_sha256(PILOT_RECORD),
            _relative(SCORES_TABLE): file_sha256(SCORES_TABLE),
            _relative(CROP_LINEAGE): file_sha256(CROP_LINEAGE),
        },
        "c2_status": "DEFERRED",
        "c2_note": (
            "No deployment risk control, UCB method, or operating threshold is frozen "
            "here. This stage only records the development statistics C2 will need."
        ),
    }
    _write_json_once(C3_CERTIFICATE, certificate)
    print(f"C3: {verdict}" + (f" (failed: {failed})" if failed else ""))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--evidence", action="store_true")
    action.add_argument("--pilot", action="store_true")
    action.add_argument("--stats", action="store_true")
    action.add_argument("--c3", action="store_true")
    args = parser.parse_args()
    try:
        if args.evidence:
            return run_evidence()
        if args.pilot:
            return run_pilot()
        if args.stats:
            return run_stats()
        return run_c3()
    except PilotError as error:
        print(f"SGV1 PILOT ERROR: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
