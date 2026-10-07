#!/usr/bin/env python
"""Freeze SGV1 OCR-only development sites and natural candidates before GT access.

Execution order is deliberately split into three commands::

    --write-plan   bind the code, rules, inputs, roles, and TRAIN-only fit policy
    --sites        enumerate and freeze the OCR-only site table
    --candidates   generate and freeze the primary g8_union candidate stream

``--audit`` verifies the frozen artifacts without loading annotations.  With
``--rederive`` it also regenerates both tables in memory and proves semantic equality.
The script never imports an annotation, alignment, outcome, or dataset loader.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from ocr_risk.canonical import rebuild_stream
from ocr_risk.discovery.enumerator import DiscoveryRules
from ocr_risk.experiments.cgv3_confirmatory import (
    CONTEXT_CHARS,
    discovery_pass,
    fit_fold_resources,
    generation_pass,
)
from ocr_risk.experiments.sgv1_frames import (
    CandidateFreeze,
    freeze_candidates,
    label_columns_present,
)
from ocr_risk.experiments.sgv1_reserve import (
    CALIBRATION,
    CONFIRMATORY,
    DEVELOPMENT,
    TRAIN,
    assert_sgv1_development_access,
    load_reserve_lock,
    load_role_manifest,
)
from ocr_risk.io.hashing import canonical_hash, file_sha256, stable_string_set_hash
from ocr_risk.schemas.spans import CanonicalSpan

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "manifests/sgv1/development_candidate_plan.json"
PLAN_AMENDMENTS = (
    REPO / "manifests/sgv1/development_candidate_plan_amendment_001.json",
    REPO / "manifests/sgv1/development_candidate_plan_amendment_002.json",
)
# Amendment 001 alone carries the site-table round-trip normalization; later links in the
# chain amend code, not that hash.
SITE_SEMANTIC_AMENDMENT = PLAN_AMENDMENTS[0]
ROLE_MANIFEST = REPO / "manifests/sgv1/role_manifest.json"
RESERVE_LOCK = REPO / "manifests/sgv1/confirmatory_reserve_lock.json"
RESERVE_SNAPSHOT = REPO / "results/generated/sgv1/reserve/pre_access_freshness_snapshot.json"
C1_CERTIFICATE = REPO / "results/generated/sgv1/corpus_qualification/c1_cord_certificate.json"
OCR_FREEZE = REPO / "results/generated/sgv1/dev_ocr/canonical_ocr_freeze.json"
OCR_SPANS = REPO / "results/generated/sgv1/dev_ocr/canonical_spans.parquet"
OUT_DIR = REPO / "results/generated/sgv1/dev_candidates"
SITE_TABLE = OUT_DIR / "sites_pre_gt.parquet"
SITE_FREEZE = OUT_DIR / "site_freeze.json"
LADDER_TABLE = OUT_DIR / "candidate_ladder_pre_gt.parquet"
CANDIDATE_TABLE = OUT_DIR / "candidates_pre_gt.parquet"
CANDIDATE_FREEZE = OUT_DIR / "candidate_freeze.json"

ENGINES = ("doctr", "easyocr", "paddleocr", "tesseract")
ENUMERATION_ROLES = (TRAIN, CALIBRATION, DEVELOPMENT)
PRIMARY_GENERATOR = "g8_union"
UNION_CAP = 4
SCHEMA = "sgv1-development-candidate-plan-v1"

CODE_BINDINGS = (
    "scripts/sgv1_development_candidates.py",
    "src/ocr_risk/discovery/enumerator.py",
    "src/ocr_risk/discovery/views.py",
    "src/ocr_risk/candidates/edit_aware.py",
    "src/ocr_risk/candidates/lexical.py",
    "src/ocr_risk/candidates/structural_v2.py",
    "src/ocr_risk/experiments/cgv3_confirmatory.py",
    "src/ocr_risk/experiments/cgv3_track_a.py",
    "src/ocr_risk/experiments/sgv1_frames.py",
    "src/ocr_risk/experiments/sgv1_reserve.py",
)


class DevelopmentCandidateError(RuntimeError):
    """An upstream freeze, GT-blindness, or reserve invariant failed."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DevelopmentCandidateError(f"expected JSON object at {path}")
    return payload


def _relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise DevelopmentCandidateError(
            f"refusing to overwrite frozen or partial artifact {path}; audit it first"
        ) from error


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DevelopmentCandidateError(
            f"refusing to overwrite frozen or partial artifact {path}; audit it first"
        )
    temporary = path.with_suffix(f"{path.suffix}.partial")
    if temporary.exists():
        raise DevelopmentCandidateError(
            f"preserved partial artifact exists at {temporary}; investigate before retrying"
        )
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _code_hashes() -> dict[str, str]:
    return {relative: file_sha256(REPO / relative) for relative in CODE_BINDINGS}


def _locked_roles() -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate the lock and guard ids before any OCR table path is resolved."""
    load_reserve_lock(
        RESERVE_LOCK,
        role_manifest_path=ROLE_MANIFEST,
        snapshot_path=RESERVE_SNAPSHOT,
    )
    manifest = load_role_manifest(ROLE_MANIFEST)
    role_of = manifest["role_of"]
    selected = sorted(document_id for document_id, role in role_of.items() if role != CONFIRMATORY)
    fit_documents = sorted(document_id for document_id, role in role_of.items() if role == TRAIN)
    assert_sgv1_development_access(
        selected,
        operation="SGV1 OCR-only site discovery and candidate generation",
        role_manifest_path=ROLE_MANIFEST,
        lock_path=RESERVE_LOCK,
        snapshot_path=RESERVE_SNAPSHOT,
    )
    return manifest, selected, fit_documents


def _plan_payload() -> dict[str, Any]:
    manifest, selected, fit_documents = _locked_roles()
    certificate = _read_json(C1_CERTIFICATE)
    if certificate.get("status") != "C1_PASS":
        raise DevelopmentCandidateError("candidate plan requires the canonical C1_PASS")
    ocr_freeze = _read_json(OCR_FREEZE)
    if not ocr_freeze.get("frozen") or ocr_freeze.get("confirmatory_accessed"):
        raise DevelopmentCandidateError("canonical development OCR freeze is not admissible")
    if ocr_freeze.get("gt_or_label_columns"):
        raise DevelopmentCandidateError("canonical OCR freeze reports GT/label columns")
    if file_sha256(OCR_SPANS) != ocr_freeze.get("canonical_spans_sha256"):
        raise DevelopmentCandidateError("canonical OCR parquet does not match its freeze")
    return {
        "schema_version": SCHEMA,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git_head(),
        "scientific_scope": "SGV1 development upstream freeze only",
        "analysis_roles": list(ENUMERATION_ROLES),
        "confirmatory_accessed": False,
        "ground_truth_allowed": False,
        "selection": {
            "document_count": len(selected),
            "document_set_sha256": stable_string_set_hash(selected),
            "pair_count": len(selected) * len(ENGINES),
            "engines": list(ENGINES),
        },
        "resource_fit": {
            "roles": [TRAIN],
            "document_count": len(fit_documents),
            "document_set_sha256": stable_string_set_hash(fit_documents),
            "held_out_engine_rule": "(E \\ {held_out}) x TRAIN OCR streams",
            "ground_truth_allowed": False,
        },
        "site_discovery": {
            "implementation": "corrected CGV3 OCR-only OcrPageView + enumerate_sites",
            "rules": asdict(DiscoveryRules()),
            "empty_pair_policy": "retain and enumerate in site freeze; never silently drop",
        },
        "candidate_generation": {
            "full_audit_ladder": [
                "b0_conf_only",
                "g3_edit_aware",
                "g7_structural_v2",
                PRIMARY_GENERATOR,
            ],
            "primary_natural_stream": PRIMARY_GENERATOR,
            "union_order": ["g3_edit_aware", "g7_structural_v2"],
            "union_cap": UNION_CAP,
            "deduplication_unit": "site_id + candidate_text, incumbent first",
            "identity_policy": "candidate_text == original_ocr is excluded",
            "candidate_id_policy": "sha256 over frozen site/edit identity",
        },
        "freeze_order": [
            "canonical_ocr",
            "sites_pre_gt",
            "candidates_pre_gt",
            "labels_after_freeze_only",
        ],
        "inputs": {
            _relative(C1_CERTIFICATE): file_sha256(C1_CERTIFICATE),
            _relative(ROLE_MANIFEST): file_sha256(ROLE_MANIFEST),
            _relative(RESERVE_LOCK): file_sha256(RESERVE_LOCK),
            _relative(RESERVE_SNAPSHOT): file_sha256(RESERVE_SNAPSHOT),
            _relative(OCR_FREEZE): file_sha256(OCR_FREEZE),
            _relative(OCR_SPANS): file_sha256(OCR_SPANS),
        },
        "code_sha256": _code_hashes(),
        "role_counts": manifest["counts"],
        "forbidden_inputs": [
            "annotations",
            "ground truth",
            "alignments",
            "candidate outcome labels",
            "verifier scores",
            "confirmatory document content",
        ],
    }


def write_plan() -> int:
    _write_json_once(PLAN, _plan_payload())
    print(f"candidate plan frozen: {_relative(PLAN)} sha256={file_sha256(PLAN)}")
    return 0


def validate_plan() -> dict[str, Any]:
    if not PLAN.is_file():
        raise DevelopmentCandidateError("candidate plan is absent; run --write-plan first")
    plan = _read_json(PLAN)
    if plan.get("schema_version") != SCHEMA:
        raise DevelopmentCandidateError("unsupported development candidate plan schema")
    if plan.get("confirmatory_accessed") or plan.get("ground_truth_allowed"):
        raise DevelopmentCandidateError("candidate plan does not enforce GT/reserve blindness")
    manifest, selected, fit_documents = _locked_roles()
    expected_selection = plan.get("selection", {})
    if expected_selection.get("document_count") != len(selected):
        raise DevelopmentCandidateError("candidate plan document count drifted")
    if expected_selection.get("document_set_sha256") != stable_string_set_hash(selected):
        raise DevelopmentCandidateError("candidate plan document set drifted")
    expected_fit = plan.get("resource_fit", {})
    if expected_fit.get("roles") != [TRAIN]:
        raise DevelopmentCandidateError("candidate resources are not frozen to TRAIN only")
    if expected_fit.get("document_set_sha256") != stable_string_set_hash(fit_documents):
        raise DevelopmentCandidateError("candidate TRAIN resource-fit set drifted")
    if plan.get("role_counts") != manifest["counts"]:
        raise DevelopmentCandidateError("role counts drifted since candidate plan")
    if plan.get("site_discovery", {}).get("rules") != asdict(DiscoveryRules()):
        raise DevelopmentCandidateError("CGV3 discovery rules drifted since candidate plan")
    if plan.get("candidate_generation", {}).get("union_cap") != UNION_CAP:
        raise DevelopmentCandidateError("candidate union cap drifted since candidate plan")
    for relative, expected in plan.get("inputs", {}).items():
        path = REPO / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise DevelopmentCandidateError(f"frozen candidate input moved: {relative}")
    current_code = _code_hashes()
    if plan.get("code_sha256") != current_code:
        _validate_amendment_chain(plan, current_code)
    return plan


def _validate_amendment_chain(plan: dict[str, Any], current_code: dict[str, str]) -> None:
    """Walk the recorded amendments from the plan's code state to the current one.

    A chain rather than a single amendment because the plan binds ten files, some of
    which are shared with later SGV1 stages: post-freeze work on a *downstream* stage can
    move a bound file without touching anything the freeze path executes. A single-slot
    amendment turns that into a permanently unrunnable audit tool, which is how defect
    SGV1-R1 was found -- ``sgv1_frames.py`` drifted in the GT-labeling commit and every
    verb of this script, ``--audit`` included, had been failing closed ever since.

    Each link must bind this plan, start where the previous link ended, and attest to the
    artifacts that existed when it was issued. The guard is not weakened: the current code
    must still equal the last link's recorded hashes exactly.
    """
    present = [path for path in PLAN_AMENDMENTS if path.is_file()]
    if not present:
        raise DevelopmentCandidateError("candidate/discovery code moved since the plan freeze")

    previous = plan.get("code_sha256")
    for path in present:
        amendment = _read_json(path)
        if amendment.get("schema_version") != "sgv1-development-candidate-plan-amendment-v1":
            raise DevelopmentCandidateError(f"unsupported amendment schema: {path.name}")
        if amendment.get("status") != "SCOPED_IMPLEMENTATION_AMENDMENT":
            raise DevelopmentCandidateError(f"amendment is not active: {path.name}")
        if amendment.get("plan_sha256") != file_sha256(PLAN):
            raise DevelopmentCandidateError(f"amendment binds another plan: {path.name}")
        if amendment.get("previous_code_sha256") != previous:
            raise DevelopmentCandidateError(
                f"amendment does not bind the preceding code state: {path.name}"
            )
        if amendment.get("site_freeze_sha256") != file_sha256(SITE_FREEZE):
            raise DevelopmentCandidateError(f"amendment binds another site freeze: {path.name}")
        if amendment.get("site_table_sha256") != file_sha256(SITE_TABLE):
            raise DevelopmentCandidateError(f"amendment binds another site table: {path.name}")
        # Each link attests to the artifacts that existed when it was issued: a
        # pre-candidate amendment attests that none did, a post-candidate one must pin the
        # candidate outputs and prove they did not move.
        if amendment.get("candidate_outputs_present_at_amendment") is False:
            if CANDIDATE_TABLE.is_file() and path is present[-1]:
                raise DevelopmentCandidateError(
                    f"{path.name} attests pre-candidate timing but candidates now exist"
                )
        elif amendment.get("candidate_outputs_present_at_amendment") is True:
            for label, artifact in (
                ("candidate_freeze_sha256", CANDIDATE_FREEZE),
                ("candidate_table_sha256", CANDIDATE_TABLE),
            ):
                if amendment.get(label) != file_sha256(artifact):
                    raise DevelopmentCandidateError(
                        f"{path.name} binds another {label.removesuffix('_sha256')}"
                    )
        else:
            raise DevelopmentCandidateError(f"amendment does not attest its timing: {path.name}")
        previous = amendment.get("amended_code_sha256")

    if previous != current_code:
        raise DevelopmentCandidateError("candidate code moved beyond the scoped amendments")


def _spans_and_streams() -> tuple[
    dict[tuple[str, str], list[CanonicalSpan]], dict[tuple[str, str], str], dict[str, str]
]:
    """Guard selection before resolving and reading the canonical OCR parquet."""
    manifest, selected, _ = _locked_roles()
    plan = validate_plan()
    assert_sgv1_development_access(
        selected,
        operation="SGV1 canonical OCR load for site/candidate freeze",
        role_manifest_path=ROLE_MANIFEST,
        lock_path=RESERVE_LOCK,
        snapshot_path=RESERVE_SNAPSHOT,
    )
    canonical_path = OCR_SPANS
    table = pq.read_table(canonical_path)
    leaking = label_columns_present(table.column_names)
    if leaking:
        raise DevelopmentCandidateError(f"canonical OCR table has GT/label columns {leaking}")
    selected_set = set(selected)
    roles = {str(key): str(value) for key, value in manifest["role_of"].items()}
    grouped = {(document_id, engine): [] for document_id in selected for engine in ENGINES}
    seen_rows = 0
    for raw in table.to_pylist():
        document_id = str(raw["document_id"])
        engine_id = str(raw["engine_id"])
        if document_id not in selected_set:
            raise DevelopmentCandidateError(
                f"canonical OCR table contains unselected/reserve id {document_id}"
            )
        if engine_id not in ENGINES:
            raise DevelopmentCandidateError(f"canonical OCR table has unknown engine {engine_id}")
        grouped[(document_id, engine_id)].append(CanonicalSpan.model_validate(raw))
        seen_rows += 1
    ocr_freeze = _read_json(OCR_FREEZE)
    if seen_rows != ocr_freeze["canonical_span_count"]:
        raise DevelopmentCandidateError("canonical OCR row census drifted")
    if len(grouped) != plan["selection"]["pair_count"]:
        raise DevelopmentCandidateError("canonical OCR pair matrix drifted")
    for spans in grouped.values():
        spans.sort(key=lambda span: span.reading_order)
    streams = {pair: rebuild_stream(spans) for pair, spans in grouped.items() if spans}
    return grouped, streams, roles


def _fit_resources(
    streams: dict[tuple[str, str], str], roles: dict[str, str]
) -> tuple[dict[str, Any], frozenset[str]]:
    fit_documents = frozenset(document_id for document_id, role in roles.items() if role == TRAIN)
    return fit_fold_resources(streams, fit_documents, ENGINES), fit_documents


def _frame_hash(frame: pd.DataFrame) -> str:
    # Parquet round-trips nullable object columns as None, while a DataFrame assembled
    # from sparse dictionaries represents the same missing cells as float NaN. Pandas
    # correctly regards the frames as equal, but a provenance hash must normalize that
    # representation explicitly or a valid write can never be consumed (amendment 001).
    normalized = frame.astype(object).where(pd.notna(frame), None)
    return canonical_hash({"columns": list(frame.columns), "rows": normalized.to_dict("records")})


def _expected_site_semantic_sha256(site_record: dict[str, Any]) -> str:
    if not SITE_SEMANTIC_AMENDMENT.is_file():
        return str(site_record["site_table_semantic_sha256"])
    amendment = _read_json(SITE_SEMANTIC_AMENDMENT)
    if amendment.get("site_freeze_sha256") != file_sha256(SITE_FREEZE):
        raise DevelopmentCandidateError("site semantic amendment binds another freeze")
    return str(amendment["site_table_roundtrip_semantic_sha256"])


def _ensure_gt_blind(frame: pd.DataFrame, *, context: str) -> None:
    leaking = label_columns_present(frame.columns)
    if leaking:
        raise DevelopmentCandidateError(f"{context} contains GT/label columns {leaking}")


def _ensure_development_ids(frame: pd.DataFrame, *, operation: str) -> None:
    if "document_id" not in frame.columns:
        raise DevelopmentCandidateError(f"{operation} table has no document_id")
    document_ids = sorted(frame["document_id"].astype(str).unique())
    assert_sgv1_development_access(
        document_ids,
        operation=operation,
        role_manifest_path=ROLE_MANIFEST,
        lock_path=RESERVE_LOCK,
        snapshot_path=RESERVE_SNAPSHOT,
    )


def _discover() -> tuple[pd.DataFrame, list[tuple[str, str]], dict[str, Any]]:
    spans, streams, roles = _spans_and_streams()
    fitted, fit_documents = _fit_resources(streams, roles)
    sites, empty = discovery_pass(
        spans,
        fitted,
        ENGINES,
        DiscoveryRules(),
        progress=print,
    )
    if sites.empty:
        raise DevelopmentCandidateError("OCR-only enumerator produced no sites")
    sites.insert(4, "role", sites["document_id"].map(roles))
    sites = sites.sort_values(
        ["document_id", "engine_id", "char_start", "char_end", "site_id"],
        kind="stable",
    ).reset_index(drop=True)
    _ensure_gt_blind(sites, context="site table")
    _ensure_development_ids(sites, operation="SGV1 site freeze")
    if sites["role"].isna().any() or set(sites["role"]) - set(ENUMERATION_ROLES):
        raise DevelopmentCandidateError("site table has missing or inadmissible role values")
    resources = {
        "fit_document_count": len(fit_documents),
        "fit_document_set_sha256": stable_string_set_hash(fit_documents),
        "fold_lexicon_sha256": {engine: fitted[engine].lexicon_sha for engine in ENGINES},
        "fold_bigram_total": {engine: fitted[engine].bigram_total for engine in ENGINES},
    }
    return sites, empty, resources


def freeze_sites() -> int:
    plan = validate_plan()
    if SITE_FREEZE.exists() or CANDIDATE_FREEZE.exists():
        raise DevelopmentCandidateError("site/candidate freeze already exists; use --audit")
    started = time.monotonic()
    sites, empty, resources = _discover()
    _write_parquet_once(SITE_TABLE, sites)
    record = {
        "schema_version": "sgv1-development-site-freeze-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git_head(),
        "frozen": True,
        "ground_truth_loaded": False,
        "confirmatory_accessed": False,
        "scientific_scope": "TRAIN + CALIBRATION + DEVELOPMENT only",
        "plan_path": _relative(PLAN),
        "plan_sha256": file_sha256(PLAN),
        "canonical_ocr_sha256": plan["inputs"][_relative(OCR_SPANS)],
        "site_table_path": _relative(SITE_TABLE),
        "site_table_sha256": file_sha256(SITE_TABLE),
        "site_table_semantic_sha256": _frame_hash(sites),
        "site_columns": list(sites.columns),
        "gt_or_label_columns": label_columns_present(sites.columns),
        "site_count": len(sites),
        "site_count_by_role": {
            str(key): int(value) for key, value in sites.groupby("role").size().items()
        },
        "site_count_by_engine": {
            str(key): int(value) for key, value in sites.groupby("engine_id").size().items()
        },
        "site_count_by_type": {
            str(key): int(value) for key, value in sites.groupby("site_type").size().items()
        },
        "pair_count": plan["selection"]["pair_count"],
        "pairs_with_sites": int(sites[["document_id", "engine_id"]].drop_duplicates().shape[0]),
        "empty_pairs": [f"{document_id}:{engine}" for document_id, engine in empty],
        "resource_fit": resources,
        "rules": asdict(DiscoveryRules()),
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_once(SITE_FREEZE, record)
    print(
        f"sites frozen: {len(sites)} sites, {len(empty)} empty pairs, "
        f"sha256={record['site_table_sha256']}"
    )
    return 0


def _source_lookup(ladder: pd.DataFrame) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for generator_id in ("g3_edit_aware", "g7_structural_v2"):
        subset = ladder[ladder["generator_id"] == generator_id]
        for row in subset.itertuples(index=False):
            lookup.setdefault((str(row.site_id), str(row.candidate_text)), generator_id)
    return lookup


def _candidate_id(row: pd.Series) -> str:
    digest = canonical_hash(
        {
            "schema": "sgv1-natural-candidate-id-v1",
            "site_id": str(row["site_id"]),
            "document_id": str(row["document_id"]),
            "engine_id": str(row["engine_id"]),
            "generator_id": str(row["generator_id"]),
            "candidate_text": str(row["candidate_text"]),
            "operation": str(row["operation"]),
        }
    )
    return f"sgv1-candidate-{digest}"


def _generate(sites: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spans, streams, roles = _spans_and_streams()
    fitted, fit_documents = _fit_resources(streams, roles)
    ladder = generation_pass(spans, streams, sites, fitted, ENGINES, progress=print)
    if ladder.empty:
        raise DevelopmentCandidateError("frozen sites produced no candidate proposals")
    ladder = ladder.sort_values(
        ["document_id", "engine_id", "site_id", "generator_id", "generator_rank"],
        kind="stable",
    ).reset_index(drop=True)
    _ensure_gt_blind(ladder, context="candidate ladder")
    _ensure_development_ids(ladder, operation="SGV1 candidate-ladder freeze")

    primary = ladder[ladder["generator_id"] == PRIMARY_GENERATOR].copy()
    if primary.empty:
        raise DevelopmentCandidateError("g8_union produced no natural candidates")
    source_lookup = _source_lookup(ladder)
    primary["generator_source"] = [
        source_lookup.get((str(row.site_id), str(row.candidate_text)), "")
        for row in primary.itertuples(index=False)
    ]
    if (primary["generator_source"] == "").any():
        raise DevelopmentCandidateError("a g8_union candidate has no auditable component source")

    site_columns = [
        "site_id",
        "role",
        "anchor_kind",
        "anchor_ref",
        "char_start",
        "char_end",
        "site_type",
        "suspicion_score",
        "provenance_reason",
    ]
    primary = primary.merge(sites[site_columns], on="site_id", how="left", validate="many_to_one")
    primary["original_ocr"] = primary["region_ocr"]
    contexts: dict[str, tuple[str, str]] = {}
    for site in sites.itertuples(index=False):
        stream = streams.get((str(site.document_id), str(site.engine_id)), "")
        start = int(site.char_start)
        end = int(site.char_end)
        contexts[str(site.site_id)] = (
            stream[max(0, start - CONTEXT_CHARS) : start],
            stream[end : end + CONTEXT_CHARS],
        )
    primary["context_before"] = [contexts[str(site_id)][0] for site_id in primary["site_id"]]
    primary["context_after"] = [contexts[str(site_id)][1] for site_id in primary["site_id"]]
    primary["candidate_id"] = primary.apply(_candidate_id, axis=1)
    if primary["candidate_id"].duplicated().any():
        raise DevelopmentCandidateError("deterministic candidate ids are not unique")
    if (primary["candidate_text"] == primary["original_ocr"]).any():
        raise DevelopmentCandidateError("identity edit reached the natural candidate stream")
    max_per_site = int(primary.groupby("site_id").size().max())
    if max_per_site > UNION_CAP:
        raise DevelopmentCandidateError(
            f"primary generator emitted {max_per_site} candidates at a site, cap is {UNION_CAP}"
        )
    columns = [
        "candidate_id",
        "site_id",
        "document_id",
        "dataset_id",
        "engine_id",
        "role",
        "generator_id",
        "generator_source",
        "candidate_text",
        "original_ocr",
        "context_before",
        "context_after",
        "operation",
        "generator_rank",
        "generator_score",
        "anchor_kind",
        "anchor_ref",
        "char_start",
        "char_end",
        "site_type",
        "suspicion_score",
        "provenance_reason",
    ]
    primary = (
        primary[columns]
        .sort_values(
            ["document_id", "engine_id", "site_id", "generator_rank", "candidate_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    _ensure_gt_blind(primary, context="primary candidate table")
    _ensure_development_ids(primary, operation="SGV1 natural-candidate freeze")
    candidate_freeze = freeze_candidates(primary, frame="natural")
    resources = {
        "fit_document_count": len(fit_documents),
        "fit_document_set_sha256": stable_string_set_hash(fit_documents),
        "fold_lexicon_sha256": {engine: fitted[engine].lexicon_sha for engine in ENGINES},
        "fold_bigram_total": {engine: fitted[engine].bigram_total for engine in ENGINES},
        "candidate_freeze": candidate_freeze,
    }
    return ladder, primary, resources


def freeze_candidate_stream() -> int:
    validate_plan()
    if not SITE_FREEZE.is_file() or not SITE_TABLE.is_file():
        raise DevelopmentCandidateError("site freeze is absent; run --sites first")
    if CANDIDATE_FREEZE.exists() or CANDIDATE_TABLE.exists() or LADDER_TABLE.exists():
        raise DevelopmentCandidateError("candidate output already exists; use --audit")
    site_record = _read_json(SITE_FREEZE)
    if not site_record.get("frozen") or site_record.get("ground_truth_loaded"):
        raise DevelopmentCandidateError("site record is not an admissible pre-GT freeze")
    if file_sha256(SITE_TABLE) != site_record.get("site_table_sha256"):
        raise DevelopmentCandidateError("site table bytes moved after their freeze")
    sites = pd.read_parquet(SITE_TABLE)
    if _frame_hash(sites) != _expected_site_semantic_sha256(site_record):
        raise DevelopmentCandidateError("site table semantics moved after their freeze")
    started = time.monotonic()
    ladder, candidates, resources = _generate(sites)
    _write_parquet_once(LADDER_TABLE, ladder)
    _write_parquet_once(CANDIDATE_TABLE, candidates)
    semantic: CandidateFreeze = resources.pop("candidate_freeze")
    record = {
        "schema_version": "sgv1-development-candidate-freeze-v1",
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_head": _git_head(),
        "execution_complete": True,
        "ground_truth_loaded": False,
        "confirmatory_accessed": False,
        "scientific_scope": "TRAIN + CALIBRATION + DEVELOPMENT only",
        "frame": "natural",
        "plan_path": _relative(PLAN),
        "plan_sha256": file_sha256(PLAN),
        "plan_amendment_path": _relative(PLAN_AMENDMENTS[0]),
        "plan_amendment_sha256": file_sha256(PLAN_AMENDMENTS[0]),
        "plan_amendments": {
            _relative(path): file_sha256(path) for path in PLAN_AMENDMENTS if path.is_file()
        },
        "site_freeze_path": _relative(SITE_FREEZE),
        "site_freeze_sha256": file_sha256(SITE_FREEZE),
        "site_table_sha256": file_sha256(SITE_TABLE),
        "ladder_table_path": _relative(LADDER_TABLE),
        "ladder_table_sha256": file_sha256(LADDER_TABLE),
        "ladder_table_semantic_sha256": _frame_hash(ladder),
        "candidate_table_path": _relative(CANDIDATE_TABLE),
        "candidate_table_file_sha256": file_sha256(CANDIDATE_TABLE),
        "candidates_sha256": semantic.candidates_sha256,
        "candidate_id_set_sha256": stable_string_set_hash(candidates["candidate_id"]),
        "candidate_columns": list(candidates.columns),
        "gt_or_label_columns": label_columns_present(candidates.columns),
        "candidate_count": len(candidates),
        "document_count_with_candidates": semantic.n_documents,
        "site_count_with_candidates": int(candidates["site_id"].nunique()),
        "ladder_candidate_count": len(ladder),
        "primary_generator": PRIMARY_GENERATOR,
        "generator_ids": list(semantic.generator_ids),
        "union_cap": UNION_CAP,
        "maximum_candidates_per_site": int(candidates.groupby("site_id").size().max()),
        "candidate_count_by_role": {
            str(key): int(value) for key, value in candidates.groupby("role").size().items()
        },
        "candidate_count_by_engine": {
            str(key): int(value) for key, value in candidates.groupby("engine_id").size().items()
        },
        "candidate_count_by_operation": {
            str(key): int(value) for key, value in candidates.groupby("operation").size().items()
        },
        "candidate_count_by_source": {
            str(key): int(value)
            for key, value in candidates.groupby("generator_source").size().items()
        },
        "resource_fit": resources,
        "elapsed_seconds": time.monotonic() - started,
        "statement": (
            "OCR-only sites and the g8_union natural candidate stream are complete and "
            "frozen. No annotation, alignment, outcome label, verifier score, or "
            "confirmatory content was loaded. Labels may enter only after this hash."
        ),
    }
    _write_json_once(CANDIDATE_FREEZE, record)
    print(
        f"candidates frozen: {len(candidates)} primary rows ({len(ladder)} ladder rows), "
        f"semantic_sha256={semantic.candidates_sha256}"
    )
    return 0


def audit(*, rederive: bool = False) -> int:
    plan = validate_plan()
    site_record = _read_json(SITE_FREEZE)
    if file_sha256(SITE_TABLE) != site_record.get("site_table_sha256"):
        raise DevelopmentCandidateError("site table file hash mismatch")
    sites = pd.read_parquet(SITE_TABLE)
    _ensure_gt_blind(sites, context="frozen site table")
    _ensure_development_ids(sites, operation="SGV1 frozen-site audit")
    if _frame_hash(sites) != _expected_site_semantic_sha256(site_record):
        raise DevelopmentCandidateError("site table semantic hash mismatch")
    if site_record.get("plan_sha256") != file_sha256(PLAN):
        raise DevelopmentCandidateError("site freeze binds another candidate plan")
    candidate_record = _read_json(CANDIDATE_FREEZE)
    if candidate_record.get("site_freeze_sha256") != file_sha256(SITE_FREEZE):
        raise DevelopmentCandidateError("candidate freeze binds another site freeze")
    if file_sha256(LADDER_TABLE) != candidate_record.get("ladder_table_sha256"):
        raise DevelopmentCandidateError("candidate ladder file hash mismatch")
    ladder = pd.read_parquet(LADDER_TABLE)
    _ensure_gt_blind(ladder, context="frozen candidate ladder")
    if _frame_hash(ladder) != candidate_record.get("ladder_table_semantic_sha256"):
        raise DevelopmentCandidateError("candidate ladder semantic hash mismatch")
    if file_sha256(CANDIDATE_TABLE) != candidate_record.get("candidate_table_file_sha256"):
        raise DevelopmentCandidateError("candidate table file hash mismatch")
    candidates = pd.read_parquet(CANDIDATE_TABLE)
    _ensure_gt_blind(candidates, context="frozen primary candidates")
    _ensure_development_ids(candidates, operation="SGV1 frozen-candidate audit")
    semantic = freeze_candidates(candidates, frame="natural")
    if semantic.candidates_sha256 != candidate_record.get("candidates_sha256"):
        raise DevelopmentCandidateError("candidate semantic freeze hash mismatch")
    if stable_string_set_hash(candidates["candidate_id"]) != candidate_record.get(
        "candidate_id_set_sha256"
    ):
        raise DevelopmentCandidateError("candidate-id set hash mismatch")
    if set(candidates["generator_id"]) != {PRIMARY_GENERATOR}:
        raise DevelopmentCandidateError("frozen natural stream contains a non-primary rung")
    if candidate_record.get("plan_sha256") != file_sha256(PLAN):
        raise DevelopmentCandidateError("candidate freeze binds another candidate plan")
    if rederive:
        derived_sites, derived_empty, _ = _discover()
        if _frame_hash(derived_sites) != _expected_site_semantic_sha256(site_record):
            raise DevelopmentCandidateError("site rederivation differs from frozen semantics")
        if [f"{document_id}:{engine}" for document_id, engine in derived_empty] != site_record.get(
            "empty_pairs"
        ):
            raise DevelopmentCandidateError("empty-pair rederivation differs from site freeze")
        derived_ladder, derived_candidates, _ = _generate(derived_sites)
        if _frame_hash(derived_ladder) != candidate_record.get("ladder_table_semantic_sha256"):
            raise DevelopmentCandidateError("candidate ladder rederivation differs")
        if freeze_candidates(derived_candidates, frame="natural").candidates_sha256 != (
            candidate_record.get("candidates_sha256")
        ):
            raise DevelopmentCandidateError("primary candidate rederivation differs")
    print(
        f"candidate audit: PASS sites={len(sites)} candidates={len(candidates)} "
        f"pairs={plan['selection']['pair_count']} rederived={rederive}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-plan", action="store_true")
    action.add_argument("--sites", action="store_true")
    action.add_argument("--candidates", action="store_true")
    action.add_argument("--audit", action="store_true")
    parser.add_argument("--rederive", action="store_true", help="rederive tables during audit")
    args = parser.parse_args()
    if args.rederive and not args.audit:
        parser.error("--rederive requires --audit")
    try:
        if args.write_plan:
            return write_plan()
        if args.sites:
            return freeze_sites()
        if args.candidates:
            return freeze_candidate_stream()
        return audit(rederive=args.rederive)
    except DevelopmentCandidateError as error:
        print(f"SGV1 DEVELOPMENT CANDIDATE ERROR: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
