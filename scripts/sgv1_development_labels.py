#!/usr/bin/env python3
"""Label the frozen SGV1 development candidates, then construct the dual frames.

This is the first SGV1 stage that may read ground truth, and it earns that by
verification, not convention: every command below re-checks the pre-GT candidate
freeze (file hashes, semantic freeze, GT-column absence, reserve guard) before a
single annotation is opened. GT is then used only to attach outcome labels::

    --labels     region ground truth + outcome labels + natural census
    --reconcile  bind an unbound label table that reproduces exactly (incident SGV1-L2)
    --audit      read-only proof that frozen candidates + GT reproduce the frozen labels
    --frames     Frame A (stratified), matched pairs, Frame B (natural stream)

Labels never flow back into the frozen candidate table; frames carry them only
downstream of the freeze hash that ``attach_labels`` enforces.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ocr_risk.align import align_document
from ocr_risk.canonical import rebuild_stream
from ocr_risk.config.models import AlignmentConfig
from ocr_risk.datasets.cord import CordDataset
from ocr_risk.edits.outcome import is_harmful
from ocr_risk.experiments.cgv3_study import AlignmentIndex, label_candidate
from ocr_risk.experiments.sgv1_frames import (
    OUTCOME_UNRESOLVED,
    attach_labels,
    build_frame_a,
    build_frame_b,
    build_matched_pairs,
    evaluation_class,
    freeze_candidates,
    label_columns_present,
)
from ocr_risk.experiments.sgv1_reserve import (
    assert_sgv1_development_access,
    load_reserve_lock,
    load_role_manifest,
)
from ocr_risk.io.hashing import canonical_hash, file_sha256, stable_string_set_hash
from ocr_risk.schemas.enums import AlignmentStatus, AnchorKind, HarmPolicy
from ocr_risk.schemas.spans import CanonicalSpan

REPO = Path(__file__).resolve().parents[1]
ROLE_MANIFEST = REPO / "manifests/sgv1/role_manifest.json"
RESERVE_LOCK = REPO / "manifests/sgv1/confirmatory_reserve_lock.json"
RESERVE_SNAPSHOT = REPO / "results/generated/sgv1/reserve/pre_access_freshness_snapshot.json"
OCR_FREEZE = REPO / "results/generated/sgv1/dev_ocr/canonical_ocr_freeze.json"
OCR_SPANS = REPO / "results/generated/sgv1/dev_ocr/canonical_spans.parquet"
CANDIDATE_DIR = REPO / "results/generated/sgv1/dev_candidates"
CANDIDATE_TABLE = CANDIDATE_DIR / "candidates_pre_gt.parquet"
CANDIDATE_FREEZE = CANDIDATE_DIR / "candidate_freeze.json"
LABEL_DIR = REPO / "results/generated/sgv1/dev_labels"
LABEL_TABLE = LABEL_DIR / "labels.parquet"
LABEL_RECORD = LABEL_DIR / "labeling_record.json"
FRAME_DIR = REPO / "results/generated/sgv1/dev_frames"

ENGINES = ("doctr", "easyocr", "paddleocr", "tesseract")
HARM_POLICY = HarmPolicy.STRICT_WORSENING
UNLABELLED_DISTANCE = -1
"""``d_before``/``d_after`` for a candidate with no established region ground truth.

Negative is impossible for an edit distance, so the sentinel cannot be mistaken for a
measurement, and it is a real value rather than a parquet null -- amendment 001 exists
because a null round-tripped into a different representation and moved a hash."""
FRAME_A_PER_DOCUMENT_CAP = 10
FRAME_A_MIN_PER_CLASS = 200
FRAME_A_SEED = 20260831


class DevelopmentLabelError(RuntimeError):
    """A freeze-binding, GT-access, or reserve invariant failed."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DevelopmentLabelError(f"expected JSON object at {path}")
    return payload


def _relative(path: Path) -> str:
    """Repo-relative inside the repo, absolute outside it.

    Records name repository artifacts, but the same builders run against temporary
    paths under test; a path outside the repo is not a provenance failure worth raising.
    """
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
        raise DevelopmentLabelError(
            f"refusing to overwrite frozen or partial artifact {path}; audit it first"
        ) from error


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DevelopmentLabelError(
            f"refusing to overwrite frozen or partial artifact {path}; audit it first"
        )
    temporary = path.with_suffix(f"{path.suffix}.partial")
    if temporary.exists():
        raise DevelopmentLabelError(
            f"preserved partial artifact exists at {temporary}; investigate before retrying"
        )
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _locked_selected_ids() -> list[str]:
    load_reserve_lock(
        RESERVE_LOCK,
        role_manifest_path=ROLE_MANIFEST,
        snapshot_path=RESERVE_SNAPSHOT,
    )
    manifest = load_role_manifest(ROLE_MANIFEST)
    selected = sorted(
        document_id for document_id, role in manifest["role_of"].items() if role != "CONFIRMATORY"
    )
    assert_sgv1_development_access(
        selected,
        operation="SGV1 post-freeze labeling and frame construction",
        role_manifest_path=ROLE_MANIFEST,
        lock_path=RESERVE_LOCK,
        snapshot_path=RESERVE_SNAPSHOT,
    )
    return selected


def _verify_candidate_freeze() -> dict[str, Any]:
    """Every freeze binding must hold before GT may be opened."""
    record = _read_json(CANDIDATE_FREEZE)
    if record.get("execution_complete") is not True:
        raise DevelopmentLabelError("candidate freeze is not execution-complete")
    if record.get("ground_truth_loaded") or record.get("confirmatory_accessed"):
        raise DevelopmentLabelError("candidate freeze record is not GT/reserve blind")
    if file_sha256(CANDIDATE_TABLE) != record.get("candidate_table_file_sha256"):
        raise DevelopmentLabelError("candidate table bytes moved since the freeze")
    candidates = pd.read_parquet(CANDIDATE_TABLE)
    if label_columns_present(candidates.columns):
        raise DevelopmentLabelError("frozen candidate table carries GT/label columns")
    if freeze_candidates(candidates, frame="natural").candidates_sha256 != record.get(
        "candidates_sha256"
    ):
        raise DevelopmentLabelError("candidate semantic freeze hash mismatch")
    assert_sgv1_development_access(
        sorted(candidates["document_id"].astype(str).unique()),
        operation="SGV1 labeling candidate-table load",
        role_manifest_path=ROLE_MANIFEST,
        lock_path=RESERVE_LOCK,
        snapshot_path=RESERVE_SNAPSHOT,
    )
    return record


def _load_spans_and_streams(
    selected: list[str],
) -> tuple[dict[tuple[str, str], list[CanonicalSpan]], dict[tuple[str, str], str]]:
    """Canonical OCR spans and linearized streams, guarded before reading."""
    selected_set = set(selected)
    ocr_freeze = _read_json(OCR_FREEZE)
    if file_sha256(OCR_SPANS) != ocr_freeze.get("canonical_spans_sha256"):
        raise DevelopmentLabelError("canonical OCR parquet does not match its freeze")
    table = pd.read_parquet(OCR_SPANS)
    leaking = label_columns_present(table.columns)
    if leaking:
        raise DevelopmentLabelError(f"canonical OCR table has GT/label columns {leaking}")
    spans: dict[tuple[str, str], list[CanonicalSpan]] = {}
    for raw in table.to_dict("records"):
        span = CanonicalSpan.model_validate(raw)
        if span.document_id not in selected_set:
            raise DevelopmentLabelError(
                f"canonical OCR table contains unselected/reserve id {span.document_id}"
            )
        spans.setdefault((span.document_id, span.engine_id), []).append(span)
    for pair_spans in spans.values():
        pair_spans.sort(key=lambda span: span.reading_order)
    # rebuild_stream, not a naive join: the linearized stream places span text at stored
    # offsets with space-filled gaps and line separators, and char_start/char_end index
    # into exactly that reconstruction. Any other join makes every site slice wrong.
    streams = {pair: rebuild_stream(pair_spans) for pair, pair_spans in spans.items() if pair_spans}
    return spans, streams


def _ground_truth_bundles(selected: list[str]) -> dict[str, Any]:
    """CORD annotations for exactly the selected development documents."""
    expected = set(selected)
    bundles: dict[str, Any] = {}
    for split in ("train", "validation"):
        for bundle in CordDataset(split=split).documents():
            document_id = bundle.document.document_id
            if document_id in expected:
                bundles[document_id] = bundle
    missing = sorted(expected - set(bundles))
    if missing:
        raise DevelopmentLabelError(f"{len(missing)} selected documents have no GT bundle")
    return bundles


RESOLVED_STATUS = AlignmentStatus.RESOLVED.value


def _region_truth_by_site(
    sites: pd.DataFrame,
    bundles: dict[str, Any],
    spans: dict[tuple[str, str], list[CanonicalSpan]],
) -> tuple[dict[str, tuple[str, str]], dict[str, int]]:
    """Per-site region ground truth **and its resolution status**.

    Text convention follows CGV3 track A: a two-span GAP anchor reads the GT of the
    OCR-empty components between its flanks, every other anchor kind reads the GT of the
    components covering its spans.

    The status gate is the part CGV3 track A did not need and SGV1 does. ``AlignmentStatus``
    states the contract directly -- only RESOLVED components may produce evaluated
    correction sites, and AMBIGUOUS components are "kept and counted, excluded from
    labeling/evaluation". Without the gate an empty return is three different things at
    once: the ground truth genuinely has nothing here, the covering component is
    ambiguous or fell outside the annotated region, or the gap flanks are not ordered in
    the alignment. Only the first is a ground truth. Labeling the other two against the
    empty string asserts that the OCR is maximally wrong at a region the corpus never
    annotated, which manufactures beneficial edits out of missing annotation
    (incident SGV1-L3).
    """
    alignment_config = AlignmentConfig()
    indexes: dict[tuple[str, str], AlignmentIndex] = {}
    status_of: dict[tuple[tuple[str, str], str], str] = {}
    diagnostics = {"pairs_aligned": 0, "sites_index_missing": 0}
    for pair in sorted({(str(row.document_id), str(row.engine_id)) for row in sites.itertuples()}):
        document_id, _engine_id = pair
        pair_spans = spans.get(pair, [])
        tokens = bundles[document_id].gt_tokens
        outcome = align_document(
            bundles[document_id].document, pair_spans, tokens, alignment_config
        )
        records = [record.model_dump(mode="json") for record in outcome.records]
        alignment_frame = pd.DataFrame(records)
        token_frame = pd.DataFrame([token.model_dump(mode="json") for token in tokens])
        indexes.update(AlignmentIndex.from_frames(alignment_frame, token_frame))
        for record in records:
            status_of[pair, str(record["alignment_id"])] = str(record["status"])
        diagnostics["pairs_aligned"] += 1

    gap_value = AnchorKind.GAP.value
    truth: dict[str, tuple[str, str]] = {}
    by_span: dict[tuple[str, str], dict[str, CanonicalSpan]] = {
        pair: {span.span_id: span for span in pair_spans} for pair, pair_spans in spans.items()
    }
    for row in sites.itertuples():
        pair = (str(row.document_id), str(row.engine_id))
        site_id = str(row.site_id)
        index = indexes.get(pair)
        anchor_ids = [part for part in str(row.anchor_ref).split("\0")[1:] if part]
        if index is None:
            diagnostics["sites_index_missing"] += 1
            truth[site_id] = ("", "unresolved:no_alignment_index")
            continue
        lookup = by_span.get(pair, {})
        ordered = [span_id for span_id in anchor_ids if span_id in lookup]
        blocked = _anchor_status_block(index, status_of, pair, ordered, len(anchor_ids))
        if blocked is not None:
            truth[site_id] = ("", blocked)
            continue
        if str(row.anchor_kind) == gap_value and len(anchor_ids) == 2:
            truth[site_id] = _gap_truth_with_status(index, status_of, pair, anchor_ids)
        else:
            truth[site_id] = (index.region_truth(ordered)[1], RESOLVED_STATUS)
    return truth, diagnostics


def _anchor_status_block(
    index: AlignmentIndex,
    status_of: dict[tuple[tuple[str, str], str], str],
    pair: tuple[str, str],
    ordered_span_ids: list[str],
    declared_span_count: int,
) -> str | None:
    """The reason this anchor may not be labelled, or ``None`` if it may."""
    if len(ordered_span_ids) != declared_span_count:
        return "unresolved:span_missing_from_page"
    for span_id in ordered_span_ids:
        component = index.component_for_span(span_id)
        if component is None:
            return "unresolved:span_unaligned"
        status = status_of.get((pair, component.alignment_id), AlignmentStatus.UNRESOLVED.value)
        if status != RESOLVED_STATUS:
            return f"unresolved:{status}"
    return None


def _gap_truth_with_status(
    index: AlignmentIndex,
    status_of: dict[tuple[tuple[str, str], str], str],
    pair: tuple[str, str],
    anchor_ids: list[str],
) -> tuple[str, str]:
    """Gap ground truth, refused unless the interval it is read from is trustworthy.

    ``gap_truth`` returns the empty string both when the ground truth says nothing is
    missing between the flanks and when the flanks are not ordered in the alignment.
    Those are opposite statements, so the interval is located here and the second case
    is reported as unresolved rather than as a ground truth of "nothing".
    """
    positions = [
        next(
            (
                i
                for i, component in enumerate(index.components)
                if span_id in component.ocr_span_ids
            ),
            None,
        )
        for span_id in anchor_ids
    ]
    left, right = positions
    if left is None or right is None or left >= right:
        return "", "unresolved:gap_flanks_unordered"
    interior = [c for c in index.components[left + 1 : right] if not c.ocr_span_ids]
    if any(
        status_of.get((pair, c.alignment_id), AlignmentStatus.UNRESOLVED.value) != RESOLVED_STATUS
        for c in interior
    ):
        return "", "unresolved:gap_interior_unresolved"
    return index.gap_truth(anchor_ids[0], anchor_ids[1]), RESOLVED_STATUS


@dataclass(frozen=True, slots=True)
class LabelDerivation:
    """One deterministic pass of frozen candidates + ground truth -> outcome labels."""

    labels: pd.DataFrame
    freeze_record: dict[str, Any]
    gt_documents: tuple[str, ...]
    diagnostics: dict[str, int]
    empty_region_gt: int
    region_mismatches: int
    census: dict[str, Any]
    candidate_file_sha256: str


def _assert_candidate_freeze_unmoved(
    candidates: pd.DataFrame, freeze_record: dict[str, Any], file_sha256_before: str
) -> None:
    """Ground truth may label the frozen stream; it may never move it.

    Checked *after* labeling, not only before. A pre-flight check proves the inputs were
    frozen when the pass started; only the post-flight check proves the pass that read
    ground truth left candidate text, order, ids, and columns byte-identical.
    """
    # Column leakage first: a labelled candidate frame cannot even be re-hashed by
    # freeze_candidates, so checking it later would surface as a confusing hash error.
    leaked = label_columns_present(candidates.columns)
    if leaked:
        raise DevelopmentLabelError(f"labeling attached {leaked} to the frozen candidates")
    file_sha256_after = file_sha256(CANDIDATE_TABLE)
    if file_sha256_after != file_sha256_before:
        raise DevelopmentLabelError("frozen candidate table bytes moved during labeling")
    if file_sha256_after != freeze_record["candidate_table_file_sha256"]:
        raise DevelopmentLabelError("frozen candidate table no longer matches its freeze")
    if (
        freeze_candidates(candidates, frame="natural").candidates_sha256
        != freeze_record["candidates_sha256"]
    ):
        raise DevelopmentLabelError("candidate semantic hash moved during labeling")
    if (
        stable_string_set_hash(candidates["candidate_id"])
        != freeze_record["candidate_id_set_sha256"]
    ):
        raise DevelopmentLabelError("candidate-id set hash moved during labeling")


def _derive_labels() -> LabelDerivation:
    """Frozen candidates + ground truth -> labels. Reads frozen inputs, writes nothing.

    Writing, reconciling, and auditing all go through this one function, which is what
    makes "the artifact on disk is exactly what this code produces" a checkable claim
    rather than a convention.
    """
    selected = _locked_selected_ids()
    freeze_record = _verify_candidate_freeze()
    candidate_file_sha256 = file_sha256(CANDIDATE_TABLE)
    candidates = pd.read_parquet(CANDIDATE_TABLE)
    spans, streams = _load_spans_and_streams(selected)

    bundles = _ground_truth_bundles(selected)
    site_columns = [
        "site_id",
        "document_id",
        "engine_id",
        "anchor_kind",
        "anchor_ref",
        "char_start",
        "char_end",
    ]
    sites = candidates[site_columns].drop_duplicates(subset=["site_id"])
    truth, diagnostics = _region_truth_by_site(sites, bundles, spans)

    # The frozen original_ocr must be exactly the stream slice its site declares, or
    # the labeling region and the frozen candidate region are different objects and
    # every d_before/d_after below would be computed against the wrong text.
    stream_lookup = streams
    expected_regions = [
        stream_lookup.get((str(row.document_id), str(row.engine_id)), "")[
            int(row.char_start) : int(row.char_end)
        ]
        for row in candidates.itertuples()
    ]
    del stream_lookup
    mismatch = int((candidates["original_ocr"].astype(str) != pd.Series(expected_regions)).sum())
    if mismatch:
        raise DevelopmentLabelError(
            f"{mismatch} frozen candidates have original_ocr != stream[char_start:char_end]; "
            "the labeling region convention does not match the frozen candidate region"
        )
    label_rows: list[dict[str, Any]] = []
    empty_gt = 0
    for row in candidates.itertuples():
        region_gt, status = truth[str(row.site_id)]
        original_ocr = str(row.original_ocr)
        labelable = status == RESOLVED_STATUS
        if labelable and not region_gt:
            empty_gt += 1
        if labelable:
            d_before, d_after, accepted = label_candidate(
                original_ocr, str(row.candidate_text), region_gt
            )
            outcome_value = accepted.value
            harmful = bool(is_harmful(accepted, HARM_POLICY))
        else:
            # -1, never 0 and never null: a distance of zero is a real measurement and a
            # parquet null is the representation that moved a hash in amendment 001.
            d_before, d_after = UNLABELLED_DISTANCE, UNLABELLED_DISTANCE
            outcome_value = OUTCOME_UNRESOLVED
            harmful = False
        label_rows.append(
            {
                "candidate_id": str(row.candidate_id),
                "outcome": outcome_value,
                "is_harmful": harmful,
                "d_before": int(d_before),
                "d_after": int(d_after),
                "labelable": labelable,
                "region_gt_status": status,
                "region_is_whitespace_only": original_ocr.strip() == "",
            }
        )
    labels = pd.DataFrame(label_rows)
    if labels["candidate_id"].duplicated().any():
        raise DevelopmentLabelError("label table has duplicate candidate ids")
    _assert_unlabelled_rows_carry_no_outcome(labels)

    merged = candidates.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    merged["evaluation_class"] = merged["outcome"].map(evaluation_class)
    census = _natural_census(merged)
    _assert_candidate_freeze_unmoved(candidates, freeze_record, candidate_file_sha256)
    return LabelDerivation(
        labels=labels,
        freeze_record=freeze_record,
        gt_documents=tuple(sorted(bundles)),
        diagnostics=diagnostics,
        empty_region_gt=empty_gt,
        region_mismatches=mismatch,
        census=census,
        candidate_file_sha256=candidate_file_sha256,
    )


def _assert_unlabelled_rows_carry_no_outcome(labels: pd.DataFrame) -> None:
    """An unresolved row must not smuggle a distance, a class, or a harm flag."""
    unlabelled = labels[~labels["labelable"]]
    if not (unlabelled["outcome"] == OUTCOME_UNRESOLVED).all():
        raise DevelopmentLabelError("an unlabelled candidate carries an outcome label")
    if bool(unlabelled["is_harmful"].any()):
        raise DevelopmentLabelError("an unlabelled candidate is marked harmful")
    if not (unlabelled[["d_before", "d_after"]] == UNLABELLED_DISTANCE).all().all():
        raise DevelopmentLabelError("an unlabelled candidate carries an edit distance")
    labelled = labels[labels["labelable"]]
    if (labelled["outcome"] == OUTCOME_UNRESOLVED).any():
        raise DevelopmentLabelError("a labelled candidate carries the unresolved sentinel")
    if (labelled[["d_before", "d_after"]] < 0).any().any():
        raise DevelopmentLabelError("a labelled candidate carries a sentinel distance")


def _label_record(
    derivation: LabelDerivation, *, provenance: dict[str, Any], elapsed_seconds: float
) -> dict[str, Any]:
    """The binding record. Nothing downstream may read the labels without it."""
    labels = derivation.labels
    return {
        "schema_version": "sgv1-development-label-record-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git_head(),
        "ground_truth_loaded": True,
        "confirmatory_accessed": False,
        "gt_scope": "CORD train+validation annotations for the 710 development documents",
        "gt_document_count": len(derivation.gt_documents),
        "gt_document_set_sha256": stable_string_set_hash(derivation.gt_documents),
        "gt_policy": CordDataset.gt_policy,
        "harm_policy": HARM_POLICY.value,
        "candidate_freeze_path": _relative(CANDIDATE_FREEZE),
        "candidate_freeze_sha256": file_sha256(CANDIDATE_FREEZE),
        "candidate_table_file_sha256": derivation.freeze_record["candidate_table_file_sha256"],
        "candidates_sha256": derivation.freeze_record["candidates_sha256"],
        "candidate_id_set_sha256": derivation.freeze_record["candidate_id_set_sha256"],
        "candidate_count": int(derivation.freeze_record["candidate_count"]),
        "alignment_config": AlignmentConfig().model_dump(mode="json"),
        "alignment_code_sha256": _module_source_sha256("ocr_risk.align"),
        "outcome_code_sha256": _module_source_sha256("ocr_risk.edits.outcome"),
        "region_truth_code_sha256": _module_source_sha256("ocr_risk.experiments.cgv3_study"),
        "canonical_ocr_freeze_sha256": file_sha256(OCR_FREEZE),
        "canonical_spans_sha256": file_sha256(OCR_SPANS),
        "role_manifest_sha256": file_sha256(ROLE_MANIFEST),
        "labeling_convention": (
            "gap_truth for two-span GAP anchors, region_truth otherwise; "
            "label_candidate(original_ocr, candidate_text, region_gt)"
        ),
        "label_count": len(labels),
        "labeled_count": int(labels["labelable"].sum()),
        "excluded_unresolved_count": int((~labels["labelable"]).sum()),
        "excluded_by_reason": {
            str(k): int(v)
            for k, v in labels.loc[~labels["labelable"], "region_gt_status"].value_counts().items()
        },
        "resolved_but_empty_region_gt_count": derivation.empty_region_gt,
        "labels_table_path": _relative(LABEL_TABLE),
        "labels_table_sha256": file_sha256(LABEL_TABLE),
        "labels_semantic_sha256": _labels_semantic_sha256(labels),
        "labels_id_set_sha256": stable_string_set_hash(labels["candidate_id"]),
        "roles_labeled": sorted(derivation.census["by_role_and_class"]),
        "region_gt_diagnostics": {
            **derivation.diagnostics,
            "labelable_candidates_with_empty_region_gt": derivation.empty_region_gt,
            "original_ocr_consistency_mismatches": derivation.region_mismatches,
        },
        "eligibility_rule": (
            "Every alignment component covering a site's anchor spans -- and, for a GAP "
            "anchor, every OCR-empty component between the flanks -- must have status "
            "RESOLVED. Otherwise the candidate carries outcome 'unresolved', no distance, "
            "and no harm flag. See incidents/label_eligibility_gate_defect.json."
        ),
        "known_limitations": [
            "results/generated/sgv1/dev_labels/incidents/gap_region_whitespace_characterization.json"
        ],
        "natural_census": derivation.census,
        "provenance": provenance,
        "elapsed_seconds": elapsed_seconds,
        "statement": (
            "Ground truth entered only after the pre-GT candidate freeze. Labels attach "
            "to frozen candidate ids and never alter the frozen stream; the candidate "
            "file, semantic, and id-set hashes are re-checked after labeling."
        ),
    }


def run_labels() -> int:
    started = time.monotonic()
    derivation = _derive_labels()
    _write_parquet_once(LABEL_TABLE, derivation.labels)
    record = _label_record(
        derivation,
        provenance={
            "artifact_origin": "written by this --labels run",
            "reconciled_from_incident": None,
        },
        elapsed_seconds=time.monotonic() - started,
    )
    _write_json_once(LABEL_RECORD, record)
    print(
        f"labels written: {len(derivation.labels)} rows "
        f"(harmful={int(derivation.labels['is_harmful'].sum())}), "
        f"record={_relative(LABEL_RECORD)}"
    )
    return 0


def _natural_census(merged: pd.DataFrame) -> dict[str, Any]:
    by_outcome = {str(k): int(v) for k, v in merged["outcome"].value_counts().items()}
    by_class = {str(k): int(v) for k, v in merged["evaluation_class"].value_counts().items()}
    by_class_engine = {
        f"{cls}:{engine}": len(group)
        for (cls, engine), group in merged.groupby(["evaluation_class", "engine_id"])
    }
    docs_per_class = {
        str(cls): int(group["document_id"].nunique())
        for cls, group in merged.groupby("evaluation_class")
    }
    sites_per_class = {
        str(cls): int(group["site_id"].nunique())
        for cls, group in merged.groupby("evaluation_class")
    }
    labelled = merged[merged["labelable"]]
    clean = labelled[labelled["d_before"] == 0]
    class_sites = {cls: set(group["site_id"]) for cls, group in merged.groupby("evaluation_class")}
    natural_pair_sites = sorted(
        class_sites.get("beneficial", set()) & class_sites.get("harmful", set())
    )
    core = labelled[~labelled["region_is_whitespace_only"]]
    return {
        "candidate_count": len(merged),
        "document_count": int(merged["document_id"].nunique()),
        "site_count": int(merged["site_id"].nunique()),
        "labelable_candidate_count": len(labelled),
        "unresolved_candidate_count": int(len(merged) - len(labelled)),
        "labelable_share": round(len(labelled) / len(merged), 4) if len(merged) else None,
        "by_region_gt_status": {
            str(k): int(v) for k, v in merged["region_gt_status"].value_counts().items()
        },
        "by_region_gt_status_and_engine": {
            f"{status}:{engine}": len(group)
            for (status, engine), group in merged.groupby(["region_gt_status", "engine_id"])
        },
        "labelable_by_engine": {
            str(k): int(v) for k, v in labelled["engine_id"].value_counts().items()
        },
        "candidates_by_engine": {
            str(k): int(v) for k, v in merged["engine_id"].value_counts().items()
        },
        "by_outcome": by_outcome,
        "by_evaluation_class": by_class,
        "by_class_and_engine": by_class_engine,
        "documents_per_class": docs_per_class,
        "sites_per_class": sites_per_class,
        "clean_anchor_candidates": len(clean),
        "clean_anchor_documents": int(clean["document_id"].nunique()),
        "clean_anchor_share_of_labelable": (
            round(len(clean) / len(labelled), 4) if len(labelled) else None
        ),
        # Two denominators, always both. The selective one answers "of the edits we can
        # score, how many are harmful"; the joint one answers "of the edits the pipeline
        # actually proposes, how many are known-harmful". Reporting only the first would
        # read as a prevalence it is not, because 53.8% of the stream is unlabelable.
        "harmful_prevalence_over_labelable": (
            round(float(labelled["is_harmful"].mean()), 4) if len(labelled) else None
        ),
        "harmful_prevalence_over_all_candidates": round(float(merged["is_harmful"].mean()), 4),
        "whitespace_only_region_candidates": int(labelled["region_is_whitespace_only"].sum()),
        "non_whitespace_core_candidates": len(core),
        "non_whitespace_core_by_class": {
            str(k): int(v) for k, v in core["evaluation_class"].value_counts().items()
        },
        "non_whitespace_core_documents_per_class": {
            str(cls): int(group["document_id"].nunique())
            for cls, group in core.groupby("evaluation_class")
        },
        "overcorrection_candidates": int((labelled["outcome"] == "overcorrection").sum()),
        "overcorrection_documents": int(
            labelled.loc[labelled["outcome"] == "overcorrection", "document_id"].nunique()
        ),
        "overcorrection_sites": int(
            labelled.loc[labelled["outcome"] == "overcorrection", "site_id"].nunique()
        ),
        "natural_matched_pair_sites": len(natural_pair_sites),
        "by_role_and_class": {
            f"{role}:{cls}": len(group)
            for (role, cls), group in merged.groupby(["role", "evaluation_class"])
        },
    }


def _module_source_sha256(module_name: str) -> str:
    """Hash the source a module was imported from, package directories included.

    Recorded so a later reader can tell whether the labels were produced by this
    alignment/outcome/region-truth code or by something that has since been edited.
    """
    module = importlib.import_module(module_name)
    origin = Path(str(module.__file__))
    paths = sorted(origin.parent.rglob("*.py")) if origin.name == "__init__.py" else [origin]
    digest = hashlib.sha256()
    for candidate_path in paths:
        digest.update(candidate_path.name.encode("utf-8"))
        digest.update(candidate_path.read_bytes())
    return digest.hexdigest()


def _scalar(value: Any) -> Any:
    """Cell -> plain Python scalar. bool is tested first: it is also an int."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return str(value)


def _labels_semantic_sha256(labels: pd.DataFrame) -> str:
    """Order-sensitive content hash over explicit Python scalars.

    Casting every cell before hashing is deliberate: amendment 001 exists because a
    parquet round trip turned NaN into None and moved a hash that no scientific content
    had moved. Nothing reaches this hash as a numpy type or a parquet-specific null.
    """
    payload = {
        "columns": list(labels.columns),
        "rows": [
            [_scalar(value) for value in row] for row in labels.itertuples(index=False, name=None)
        ],
    }
    return canonical_hash(payload)


def _compare_labels(existing: pd.DataFrame, rederived: pd.DataFrame) -> dict[str, Any]:
    """Every axis a reconciliation could differ on, reported -- not one summary bool.

    A single hash comparison answers "same or not" but not "same how", and a
    reconciliation that adopts an existing artifact has to be able to say which
    properties were checked.
    """
    checks: dict[str, bool] = {
        "row_count_equal": len(existing) == len(rederived),
        "columns_equal": list(existing.columns) == list(rederived.columns),
        "dtypes_equal": [str(d) for d in existing.dtypes] == [str(d) for d in rederived.dtypes],
        "candidate_id_set_equal": (
            stable_string_set_hash(existing["candidate_id"].astype(str))
            == stable_string_set_hash(rederived["candidate_id"].astype(str))
        ),
    }
    aligned = checks["row_count_equal"] and checks["columns_equal"]
    for column in ("candidate_id", "outcome", "is_harmful", "d_before", "d_after"):
        checks[f"{column}_row_aligned_equal"] = bool(
            aligned
            and column in existing.columns
            and existing[column]
            .reset_index(drop=True)
            .equals(rederived[column].reset_index(drop=True))
        )
    existing_sha = _labels_semantic_sha256(existing)
    rederived_sha = _labels_semantic_sha256(rederived)
    checks["semantic_sha256_equal"] = existing_sha == rederived_sha
    differing = (
        int((existing.reset_index(drop=True) != rederived.reset_index(drop=True)).any(axis=1).sum())
        if aligned
        else None
    )
    return {
        "identical": all(checks.values()),
        "checks": checks,
        "existing_rows": len(existing),
        "rederived_rows": len(rederived),
        "existing_semantic_sha256": existing_sha,
        "rederived_semantic_sha256": rederived_sha,
        "differing_rows": differing,
    }


def run_reconcile() -> int:
    """Bind the unbound label table left by incident SGV1-L2 -- only if it reproduces.

    The orphan is never overwritten and never deleted. It is re-derived from the same
    frozen candidate table, the same ground truth, the same alignment configuration and
    the same code; if any comparison axis differs the run fails closed with nothing
    written, because an artifact that cannot be reproduced must not be given provenance.
    """
    started = time.monotonic()
    if LABEL_RECORD.exists():
        raise DevelopmentLabelError(
            f"{_relative(LABEL_RECORD)} already exists; there is nothing unbound to reconcile"
        )
    if not LABEL_TABLE.exists():
        raise DevelopmentLabelError(
            f"no label table at {_relative(LABEL_TABLE)}; use --labels, not --reconcile"
        )
    orphan_sha256 = file_sha256(LABEL_TABLE)
    orphan = pd.read_parquet(LABEL_TABLE)
    derivation = _derive_labels()
    comparison = _compare_labels(orphan, derivation.labels)
    if not comparison["identical"]:
        failed = sorted(name for name, ok in comparison["checks"].items() if not ok)
        raise DevelopmentLabelError(
            "rederived labels differ from the unbound artifact on "
            f"{failed}; refusing to bind it (differing_rows={comparison['differing_rows']})"
        )
    if file_sha256(LABEL_TABLE) != orphan_sha256:
        raise DevelopmentLabelError("label table bytes moved during reconciliation")
    record = _label_record(
        derivation,
        provenance={
            "artifact_origin": (
                "written by an interrupted --labels run whose record assembly raised "
                "before the binding record existed; adopted here after deterministic "
                "rederivation reproduced it exactly"
            ),
            "reconciled_from_incident": "SGV1-L2",
            "incident_record": _relative(
                LABEL_DIR / "incidents/label_record_serialization_incident.json"
            ),
            "unbound_artifact_sha256": orphan_sha256,
            "rederivation_comparison": comparison,
        },
        elapsed_seconds=time.monotonic() - started,
    )
    _write_json_once(LABEL_RECORD, record)
    print(
        f"reconciled: unbound {orphan_sha256[:12]} reproduced exactly "
        f"({len(orphan)} rows); record={_relative(LABEL_RECORD)}"
    )
    return 0


def run_audit() -> int:
    """Read-only proof that frozen candidates + GT reproduce the bound labels.

    Writes nothing, fits nothing, and never regenerates candidates or sites: it reads
    the frozen candidate table exactly as the freeze left it, re-derives the labels, and
    checks the record's bindings against the files those bindings name.
    """
    if not LABEL_RECORD.exists():
        raise DevelopmentLabelError(
            f"no binding record at {_relative(LABEL_RECORD)}; the label table is unbound "
            "and must be reconciled or written, not audited"
        )
    record = _read_json(LABEL_RECORD)
    on_disk = pd.read_parquet(LABEL_TABLE)
    bindings: dict[str, bool] = {
        "record_declares_gt_loaded": record.get("ground_truth_loaded") is True,
        "record_declares_reserve_blind": record.get("confirmatory_accessed") is False,
        "label_bytes_match_record": file_sha256(LABEL_TABLE) == record.get("labels_table_sha256"),
        "label_semantics_match_record": (
            _labels_semantic_sha256(on_disk) == record.get("labels_semantic_sha256")
        ),
        "candidate_freeze_bytes_match_record": (
            file_sha256(CANDIDATE_FREEZE) == record.get("candidate_freeze_sha256")
        ),
        "candidate_bytes_match_record": (
            file_sha256(CANDIDATE_TABLE) == record.get("candidate_table_file_sha256")
        ),
        "canonical_spans_match_record": (
            file_sha256(OCR_SPANS) == record.get("canonical_spans_sha256")
        ),
        "alignment_config_matches_record": (
            AlignmentConfig().model_dump(mode="json") == record.get("alignment_config")
        ),
        "harm_policy_matches_record": record.get("harm_policy") == HARM_POLICY.value,
        "alignment_code_matches_record": (
            _module_source_sha256("ocr_risk.align") == record.get("alignment_code_sha256")
        ),
        "outcome_code_matches_record": (
            _module_source_sha256("ocr_risk.edits.outcome") == record.get("outcome_code_sha256")
        ),
    }
    derivation = _derive_labels()
    comparison = _compare_labels(on_disk, derivation.labels)
    bindings["census_matches_record"] = derivation.census == record.get("natural_census")
    bindings["rederivation_identical"] = bool(comparison["identical"])
    failed = sorted(name for name, ok in bindings.items() if not ok)
    if failed:
        raise DevelopmentLabelError(
            f"label audit failed on {failed}; rederivation checks={comparison['checks']}"
        )
    print(
        f"label audit: PASS rows={len(on_disk)} "
        f"semantic={comparison['rederived_semantic_sha256'][:12]} "
        f"candidates={record.get('candidate_count')} rederived=True"
    )
    return 0


def _pair_side_counts(
    pairs: pd.DataFrame, pool: pd.DataFrame, side: str, column: str = "generator_source"
) -> dict[str, int]:
    """Counts of ``column`` for one side of every matched pair.

    Reported for both sides because a paired endpoint is only interpretable if the two
    members are not systematically produced by different generators: that would make
    P(q+ > q-) a generator-identification score rather than a verification score. The
    lookup runs over the whole evaluation pool, not over Frame A -- mapping through a
    sample silently drops every pair whose member was not sampled, and a partial count
    read as a full one is exactly the kind of confound this field exists to expose.
    """
    lookup = pool.set_index("candidate_id")[column]
    values = pairs[f"{side}_candidate_id"].map(lookup)
    if values.isna().any():
        msg = f"{int(values.isna().sum())} matched-pair members are outside the pool"
        raise DevelopmentLabelError(msg)
    return {str(k): int(v) for k, v in values.value_counts().items()}


def _assert_frames_are_frozen_subsets(
    candidates: pd.DataFrame,
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    pairs: pd.DataFrame,
    evaluation_pool_ids: set[str] | None = None,
) -> None:
    """No frame may contain a candidate the pre-GT freeze did not already contain.

    The failure this guards is a stratified benchmark quietly acquiring a GT-derived or
    synthetically-constructed row: Frame A samples the frozen pool, Frame B *is* the
    frozen pool, and both members of every matched pair are frozen candidates. Subset
    containment is asserted with ``<=``; ``>`` would compare in the wrong direction and
    pass on exactly the leak it was written to catch.
    """
    frozen_ids = set(candidates["candidate_id"].astype(str))
    frame_a_ids = set(frame_a["candidate_id"].astype(str))
    if not frame_a_ids <= frozen_ids:
        raise DevelopmentLabelError(
            f"Frame A carries {len(frame_a_ids - frozen_ids)} candidates outside the frozen pool"
        )
    if len(frame_b) != len(candidates) or set(frame_b["candidate_id"].astype(str)) != frozen_ids:
        raise DevelopmentLabelError("Frame B is not the exact natural candidate stream")
    pair_ids = set(pairs["plus_candidate_id"].astype(str)) | set(
        pairs["minus_candidate_id"].astype(str)
    )
    if not pair_ids <= frozen_ids:
        raise DevelopmentLabelError(
            f"matched pairs reference {len(pair_ids - frozen_ids)} unfrozen candidates"
        )
    if evaluation_pool_ids is not None and not pair_ids <= evaluation_pool_ids:
        raise DevelopmentLabelError(
            "matched pairs reference candidates outside the labelled evaluation pool"
        )
    for name, table in (("Frame A", frame_a), ("Frame B", frame_b)):
        frames = set(table["frame"].astype(str)) if "frame" in table.columns else set()
        if frames != {"natural"}:
            raise DevelopmentLabelError(
                f"{name} carries non-natural frames {sorted(frames)}; challenge and "
                "degradation material must never enter the natural stream"
            )
    for name, table in (("Frame A", frame_a), ("Frame B", frame_b), ("matched pairs", pairs)):
        assert_sgv1_development_access(
            sorted(table["document_id"].astype(str).unique()),
            operation=f"SGV1 {name} construction",
            role_manifest_path=ROLE_MANIFEST,
            lock_path=RESERVE_LOCK,
            snapshot_path=RESERVE_SNAPSHOT,
        )


def run_frames() -> int:
    started = time.monotonic()
    _locked_selected_ids()
    _verify_candidate_freeze()
    label_record = _read_json(LABEL_RECORD)
    if file_sha256(LABEL_TABLE) != label_record.get("labels_table_sha256"):
        raise DevelopmentLabelError("label table bytes moved since the labeling record")
    candidate_bytes_before = file_sha256(CANDIDATE_TABLE)
    candidates = pd.read_parquet(CANDIDATE_TABLE)
    labels = pd.read_parquet(LABEL_TABLE)
    freeze = freeze_candidates(candidates, frame="natural")

    # Frame B: the whole natural stream with labels attached, unmodified.
    labeled_full = attach_labels(candidates, freeze, labels)
    frame_b = build_frame_b(labeled_full, freeze)

    # Frame A: the DEVELOPMENT-role sub-stream is its own hash-bound frozen pool, so
    # the stratified benchmark can never contain a fitting or calibration document.
    dev_pool = candidates[candidates["role"] == "DEVELOPMENT"].reset_index(drop=True)
    dev_freeze = freeze_candidates(dev_pool, frame="natural")
    dev_labeled = attach_labels(dev_pool, dev_freeze, labels)
    frame_a, populations = build_frame_a(
        dev_labeled,
        dev_freeze,
        per_document_cap=FRAME_A_PER_DOCUMENT_CAP,
        min_per_class=FRAME_A_MIN_PER_CLASS,
        seed=FRAME_A_SEED,
    )
    # Pairs come from the whole labelled DEVELOPMENT pool, not from Frame A's sample.
    # A within-site pair only survives sampling when both members happen to be drawn,
    # which discarded 83% of the available pairs (79 sites instead of 458) for no
    # scientific reason: the paired endpoint controls for the site, and the per-document
    # cap that Frame A needs for class balance does nothing for it.
    dev_scored = dev_labeled.assign(evaluation_class=dev_labeled["outcome"].map(evaluation_class))
    dev_evaluable = dev_scored[dev_scored["labelable"]].reset_index(drop=True)
    pairs = build_matched_pairs(dev_evaluable)
    pairs_within_frame_a = build_matched_pairs(frame_a)

    _assert_frames_are_frozen_subsets(
        candidates,
        frame_a,
        frame_b,
        pairs,
        evaluation_pool_ids=set(dev_evaluable["candidate_id"].astype(str)),
    )
    if file_sha256(CANDIDATE_TABLE) != candidate_bytes_before:
        raise DevelopmentLabelError("canonical candidate table mutated during frame build")

    frame_a_full = frame_a.merge(
        # engine_id, site_id and document_id already ride along on Frame A's key
        # columns; re-merging them would produce _x/_y suffixes rather than a wider table.
        dev_labeled[
            [
                "candidate_id",
                "role",
                "generator_source",
                "site_type",
                "operation",
                "anchor_kind",
                "d_before",
                "d_after",
                "region_gt_status",
                "region_is_whitespace_only",
            ]
        ],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    if not bool((frame_a_full["region_gt_status"] == RESOLVED_STATUS).all()):
        raise DevelopmentLabelError("Frame A contains a candidate with no established region GT")
    record = {
        "schema_version": "sgv1-development-frames-record-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git_head(),
        "ground_truth_loaded": True,
        "confirmatory_accessed": False,
        "inputs": {
            _relative(CANDIDATE_FREEZE): file_sha256(CANDIDATE_FREEZE),
            _relative(CANDIDATE_TABLE): freeze.candidates_sha256,
            _relative(LABEL_RECORD): file_sha256(LABEL_RECORD),
            _relative(LABEL_TABLE): file_sha256(LABEL_TABLE),
            _relative(ROLE_MANIFEST): file_sha256(ROLE_MANIFEST),
        },
        "frame_a_parameters": {
            "population": "DEVELOPMENT-role frozen sub-stream",
            "development_pool_freeze_sha256": dev_freeze.candidates_sha256,
            "development_pool_rows": len(dev_pool),
            "per_document_cap": FRAME_A_PER_DOCUMENT_CAP,
            "min_per_class": FRAME_A_MIN_PER_CLASS,
            "seed": FRAME_A_SEED,
        },
        "frame_a": {
            "rows": len(frame_a),
            "class_populations": populations,
            "rows_by_class": {
                str(k): int(v) for k, v in frame_a["evaluation_class"].value_counts().items()
            },
            "documents_by_class": {
                str(cls): int(group["document_id"].nunique())
                for cls, group in frame_a.groupby("evaluation_class")
            },
            "sites_by_class": {
                str(cls): int(group["site_id"].nunique())
                for cls, group in frame_a.groupby("evaluation_class")
            },
            "rows_by_class_and_engine": {
                f"{cls}:{engine}": len(group)
                for (cls, engine), group in frame_a_full.groupby(["evaluation_class", "engine_id"])
            },
            "documents_by_class_and_engine": {
                f"{cls}:{engine}": int(group["document_id"].nunique())
                for (cls, engine), group in frame_a_full.groupby(["evaluation_class", "engine_id"])
            },
            "rows_by_class_and_operation": {
                f"{cls}:{operation}": len(group)
                for (cls, operation), group in frame_a_full.groupby(
                    ["evaluation_class", "operation"]
                )
            },
            "rows_by_class_and_generator_source": {
                f"{cls}:{source}": len(group)
                for (cls, source), group in frame_a_full.groupby(
                    ["evaluation_class", "generator_source"]
                )
            },
            "whitespace_only_region_rows_by_class": {
                str(cls): int(group["region_is_whitespace_only"].sum())
                for cls, group in frame_a_full.groupby("evaluation_class")
            },
            "overcorrection_rows": int((frame_a_full["outcome"] == "overcorrection").sum()),
            "overcorrection_documents": int(
                frame_a_full.loc[
                    frame_a_full["outcome"] == "overcorrection", "document_id"
                ].nunique()
            ),
            "sampling_weight_by_class": {
                str(cls): sorted({float(w) for w in group["sampling_weight"]})
                for cls, group in frame_a.groupby("evaluation_class")
            },
        },
        "matched_pairs": {
            "population": "labelled DEVELOPMENT-role frozen candidates",
            "population_rows": len(dev_evaluable),
            "pairs": len(pairs),
            "pairs_also_inside_frame_a": len(pairs_within_frame_a),
            "documents": int(pairs["document_id"].nunique()),
            "sites": int(pairs["site_id"].nunique()),
            "by_engine": {str(k): int(v) for k, v in pairs["engine_id"].value_counts().items()},
            "documents_by_engine": {
                str(engine): int(group["document_id"].nunique())
                for engine, group in pairs.groupby("engine_id")
            },
            "plus_by_generator_source": {
                str(k): int(v) for k, v in _pair_side_counts(pairs, dev_evaluable, "plus").items()
            },
            "minus_by_generator_source": {
                str(k): int(v) for k, v in _pair_side_counts(pairs, dev_evaluable, "minus").items()
            },
            "plus_by_operation": {
                str(k): int(v)
                for k, v in _pair_side_counts(pairs, dev_evaluable, "plus", "operation").items()
            },
            "minus_by_operation": {
                str(k): int(v)
                for k, v in _pair_side_counts(pairs, dev_evaluable, "minus", "operation").items()
            },
            "minus_whitespace_only_region_pairs": int(
                pairs["minus_candidate_id"]
                .map(dev_evaluable.set_index("candidate_id")["region_is_whitespace_only"])
                .sum()
            ),
            "pairs_per_document": {
                "max": int(pairs.groupby("document_id").size().max()),
                "median": float(pairs.groupby("document_id").size().median()),
                "documents_contributing_over_10_percent": int(
                    (pairs.groupby("document_id").size() > 0.10 * len(pairs)).sum()
                ),
            },
        },
        "frame_b": {
            "rows": len(frame_b),
            "documents": int(frame_b["document_id"].nunique()),
            "sites": int(frame_b["site_id"].nunique()),
            "by_class": {
                str(k): int(v) for k, v in frame_b["evaluation_class"].value_counts().items()
            },
            "labelable_rows": int(frame_b["labelable"].sum()),
            "unresolved_rows": int((~frame_b["labelable"]).sum()),
            "by_region_gt_status": {
                str(k): int(v) for k, v in frame_b["region_gt_status"].value_counts().items()
            },
            "clean_anchor_candidates": int((frame_b["d_before"] == 0).sum()),
            "erroneous_anchor_candidates": int((frame_b["d_before"] > 0).sum()),
            "clean_anchor_documents": int(
                frame_b.loc[frame_b["d_before"] == 0, "document_id"].nunique()
            ),
            "overcorrection_opportunities": int((frame_b["outcome"] == "overcorrection").sum()),
            # Both denominators, always. The unlabelable half of the stream is real
            # traffic a deployed system would still see, so the joint rate is the honest
            # lower bound and the selective rate is the honest conditional.
            "harmful_prevalence_over_labelable": round(
                float(frame_b.loc[frame_b["labelable"], "is_harmful"].mean()), 4
            ),
            "harmful_prevalence_over_all_rows": round(float(frame_b["is_harmful"].mean()), 4),
            "whitespace_only_region_rows": int(
                frame_b.loc[frame_b["labelable"], "region_is_whitespace_only"].sum()
            ),
            "by_engine": {str(k): int(v) for k, v in frame_b["engine_id"].value_counts().items()},
            "labelable_by_engine": {
                str(k): int(v)
                for k, v in frame_b.loc[frame_b["labelable"], "engine_id"].value_counts().items()
            },
            "by_class_and_engine": {
                f"{cls}:{engine}": len(group)
                for (cls, engine), group in frame_b.groupby(["evaluation_class", "engine_id"])
            },
        },
        "artifacts": {
            "frame_a_path": _relative(FRAME_DIR / "frame_a.parquet"),
            "frame_a_sha256": None,
            "evaluation_pool_path": _relative(FRAME_DIR / "evaluation_pool.parquet"),
            "evaluation_pool_sha256": None,
            "matched_pairs_path": _relative(FRAME_DIR / "matched_pairs.parquet"),
            "matched_pairs_sha256": None,
            "frame_b_path": _relative(FRAME_DIR / "frame_b.parquet"),
            "frame_b_sha256": None,
        },
        "elapsed_seconds": time.monotonic() - started,
        "statement": (
            "Frame A class proportions are a stratification design parameter and are "
            "never a prevalence statement; Frame B alone carries natural prevalence."
        ),
    }
    _write_parquet_once(FRAME_DIR / "frame_a.parquet", frame_a_full)
    _write_parquet_once(FRAME_DIR / "evaluation_pool.parquet", dev_evaluable)
    _write_parquet_once(FRAME_DIR / "matched_pairs.parquet", pairs)
    _write_parquet_once(FRAME_DIR / "frame_b.parquet", frame_b)
    record["artifacts"]["frame_a_sha256"] = file_sha256(FRAME_DIR / "frame_a.parquet")
    record["artifacts"]["evaluation_pool_sha256"] = file_sha256(
        FRAME_DIR / "evaluation_pool.parquet"
    )
    record["artifacts"]["matched_pairs_sha256"] = file_sha256(FRAME_DIR / "matched_pairs.parquet")
    record["artifacts"]["frame_b_sha256"] = file_sha256(FRAME_DIR / "frame_b.parquet")
    _write_json_once(FRAME_DIR / "frames_record.json", record)
    print(
        f"frames written: A={len(frame_a)} rows, pool={len(dev_evaluable)} rows, "
        f"{len(pairs)} pairs, B={len(frame_b)} rows; "
        f"record={_relative(FRAME_DIR / 'frames_record.json')}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--labels", action="store_true")
    action.add_argument("--reconcile", action="store_true")
    action.add_argument("--audit", action="store_true")
    action.add_argument("--frames", action="store_true")
    args = parser.parse_args()
    try:
        if args.labels:
            return run_labels()
        if args.reconcile:
            return run_reconcile()
        if args.audit:
            return run_audit()
        return run_frames()
    except DevelopmentLabelError as error:
        print(f"SGV1 DEVELOPMENT LABEL ERROR: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
