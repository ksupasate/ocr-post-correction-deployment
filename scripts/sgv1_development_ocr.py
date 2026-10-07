#!/usr/bin/env python3
"""Run, audit, and freeze SGV1 development-only OCR under the reserve lock.

Recognition is one engine per process because PaddlePaddle and PyTorch cannot safely
coexist in one macOS process.  Raw responses are write-once and fingerprint-keyed, so an
interrupted command resumes by validating and skipping completed pages.  The plan is
frozen before the first CORD train/validation image is resolved.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ocr_risk.canonical import CanonicalizationPolicy, canonicalize_response
from ocr_risk.engines import PageInput, build_engine
from ocr_risk.experiments.sgv1_reserve import (
    CALIBRATION,
    DEVELOPMENT,
    TRAIN,
    assert_sgv1_development_access,
    load_reserve_lock,
    load_role_manifest,
)
from ocr_risk.io.hashing import (
    canonical_json,
    file_sha256,
    sha256_of_bytes,
    stable_string_set_hash,
)
from ocr_risk.io.paths import data_root
from ocr_risk.io.raw_store import RawStore

REPO = Path(__file__).resolve().parents[1]
ROLE_MANIFEST = REPO / "manifests/sgv1/role_manifest.json"
LOCK = REPO / "manifests/sgv1/confirmatory_reserve_lock.json"
SNAPSHOT = REPO / "results/generated/sgv1/reserve/pre_access_freshness_snapshot.json"
C1 = REPO / "results/generated/sgv1/corpus_qualification/c1_cord_certificate.json"
ENGINE_CONFIG = REPO / "configs/engines/four_real.yaml"
PLAN = REPO / "manifests/sgv1/development_ocr_plan.json"
PLAN_AMENDMENT = REPO / "manifests/sgv1/development_ocr_plan_amendment_001.json"
OUT = REPO / "results/generated/sgv1/dev_ocr"
RUNS = OUT / "engine_runs"
INCIDENT = OUT / "incidents/tesseract_payload_validator_interruption.json"
RAW_MANIFEST = OUT / "raw_response_manifest.json"
CANONICAL_SPANS = OUT / "canonical_spans.parquet"
CANONICAL_FREEZE = OUT / "canonical_ocr_freeze.json"

ENGINE_ORDER = ("tesseract", "paddleocr", "easyocr", "doctr")
ALLOWED_ROLES = (TRAIN, CALIBRATION, DEVELOPMENT)
PLAN_SCHEMA = "sgv1-development-ocr-plan-v1"


class DevelopmentOcrError(RuntimeError):
    """An OCR action would violate the frozen plan or evidence contract."""


def _read(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_new_json(path: Path, payload: object) -> None:
    if path.exists():
        raise DevelopmentOcrError(f"refusing to replace frozen artifact {path}")
    _write_json_atomic(path, payload)


def _engine_specs() -> dict[str, dict[str, Any]]:
    config = yaml.safe_load(ENGINE_CONFIG.read_text(encoding="utf-8"))
    by_id: dict[str, dict[str, Any]] = {}
    for raw in config["engines"]:
        engine_id = str(raw["id"])
        params = dict(raw.get("params", {}))
        params.update(raw.get("dataset_params", {}).get("cord", {}))
        by_id[engine_id] = {
            "engine_id": engine_id,
            "adapter": str(raw["adapter"]),
            "params": params,
        }
    if tuple(by_id) != ENGINE_ORDER:
        raise DevelopmentOcrError(
            f"four_real engine order/set changed: {tuple(by_id)} != {ENGINE_ORDER}"
        )
    return by_id


def _locked_selection() -> tuple[dict[str, Any], list[str]]:
    load_reserve_lock(
        LOCK,
        role_manifest_path=ROLE_MANIFEST,
        snapshot_path=SNAPSHOT,
    )
    manifest = load_role_manifest(ROLE_MANIFEST)
    selected = sorted(
        document_id for document_id, role in manifest["role_of"].items() if role in ALLOWED_ROLES
    )
    assert_sgv1_development_access(
        selected,
        operation="SGV1 development OCR selection",
        role_manifest_path=ROLE_MANIFEST,
        lock_path=LOCK,
        snapshot_path=SNAPSHOT,
    )
    return manifest, selected


def _pool_rows(selected: list[str]) -> dict[str, dict[str, Any]]:
    # The guard fires before this function is called and before any path is resolved.
    certificate = _read(C1)
    if certificate.get("status") != "C1_PASS":
        raise DevelopmentOcrError(
            f"development OCR requires C1_PASS, got {certificate.get('status')}"
        )
    pool_path = REPO / certificate["inputs"]["pool"]["path"]
    if file_sha256(pool_path) != certificate["inputs"]["pool"]["sha256"]:
        raise DevelopmentOcrError("C1 pool identity changed after certification")
    pool = _read(pool_path)
    wanted = set(selected)
    rows = {
        str(row["document_id"]): row
        for row in pool["documents"]
        if str(row["document_id"]) in wanted
    }
    if set(rows) != wanted:
        missing = sorted(wanted - set(rows))
        raise DevelopmentOcrError(f"C1 identity manifest misses {len(missing)} selected ids")
    return rows


def _raw_path(store: RawStore, engine_id: str, fingerprint: str, document_id: str) -> Path:
    return store.engine_response_path("cord", engine_id, fingerprint, document_id)


def _validate_raw(
    path: Path,
    *,
    store: RawStore,
    document_id: str,
    engine_id: str,
    fingerprint: str,
    image_sha256: str,
) -> dict[str, Any]:
    raw = store.read_engine_response(path)
    expected = {
        "document_id": document_id,
        "dataset_id": "cord",
        "engine_id": engine_id,
        "engine_fingerprint": fingerprint,
    }
    observed = {key: getattr(raw, key) for key in expected}
    if observed != expected:
        raise DevelopmentOcrError(f"raw response identity mismatch at {path}: {observed}")
    if raw.payload.get("image_sha256") != image_sha256:
        raise DevelopmentOcrError(f"raw response at {path} binds the wrong source image")
    if raw.payload_format == "tesseract_tsv_v5":
        # Tesseract's adapter contract predates the structured-result adapters and pins
        # the verbatim TSV bytes. Paddle/EasyOCR/docTR pin their structured payload.
        payload_digest = sha256_of_bytes(str(raw.payload.get("tsv", "")).encode("utf-8"))
    else:
        payload_digest = sha256_of_bytes(canonical_json(raw.payload).encode("utf-8"))
    if payload_digest != raw.payload_sha256:
        raise DevelopmentOcrError(f"raw payload checksum mismatch at {path}")
    return {
        "raw_file_sha256": file_sha256(path),
        "payload_sha256": raw.payload_sha256,
        "duration_seconds": raw.duration_seconds,
        "started_at_utc": raw.started_at_utc,
        "host": raw.host,
        "platform": raw.platform,
    }


def _peak_memory_bytes() -> int:
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return maximum if sys.platform == "darwin" else maximum * 1024


def freeze_plan() -> dict[str, Any]:
    if PLAN.exists():
        return validate_plan()
    manifest, selected = _locked_selection()
    specs = _engine_specs()
    store = RawStore()
    engines: dict[str, dict[str, Any]] = {}
    for engine_id in ENGINE_ORDER:
        pointer = REPO / f"data/raw/ocr/cord/{engine_id}/fingerprint.json"
        if not pointer.is_file():
            raise DevelopmentOcrError(
                f"no historical CORD-test fingerprint pointer for {engine_id}; "
                "cannot pin instrument"
            )
        historical = _read(pointer)
        if historical.get("params") != specs[engine_id]["params"]:
            raise DevelopmentOcrError(
                f"{engine_id} CORD-test params differ from four_real.yaml: "
                f"{historical.get('params')} != {specs[engine_id]['params']}"
            )
        fingerprint = str(historical["fingerprint"])
        existing = sum(
            _raw_path(store, engine_id, fingerprint, document_id).is_file()
            for document_id in selected
        )
        if existing:
            raise DevelopmentOcrError(
                f"{engine_id} already has {existing} selected train/validation response(s); "
                "the pre-OCR plan must be issued before recognition"
            )
        engines[engine_id] = {
            **specs[engine_id],
            "expected_fingerprint": fingerprint,
            "historical_cord_test_pointer": pointer.relative_to(REPO).as_posix(),
            "historical_cord_test_pointer_sha256": file_sha256(pointer),
            "historical_cord_test_responses": len(
                list((pointer.parent / fingerprint[:8]).glob("cord-test-*.json"))
            ),
            "selected_responses_before_plan": existing,
            "device_policy": (
                "CPU; EasyOCR gpu=False; docTR predictor is never moved to an accelerator; "
                "Tesseract is an external CPU binary; PaddleOCR uses the installed "
                "macOS CPU backend"
            ),
        }
    counts = {
        role: sum(value == role for value in manifest["role_of"].values()) for role in ALLOWED_ROLES
    }
    payload = {
        "schema_version": PLAN_SCHEMA,
        "issued_utc": _utc_now(),
        "scientific_scope": "SGV1 development OCR only; CONFIRMATORY remains locked",
        "role_manifest_sha256": file_sha256(ROLE_MANIFEST),
        "reserve_lock_sha256": file_sha256(LOCK),
        "pre_access_snapshot_sha256": file_sha256(SNAPSHOT),
        "c1_certificate_sha256": file_sha256(C1),
        "allowed_roles": list(ALLOWED_ROLES),
        "document_counts_by_role": counts,
        "document_count": len(selected),
        "document_set_sha256": stable_string_set_hash(selected),
        "engine_order": list(ENGINE_ORDER),
        "engines": engines,
        "engine_config_path": ENGINE_CONFIG.relative_to(REPO).as_posix(),
        "engine_config_sha256": file_sha256(ENGINE_CONFIG),
        "runner_path": Path(__file__).resolve().relative_to(REPO).as_posix(),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "raw_storage": (
            "data/raw/ocr/cord/<engine>/<fingerprint-prefix-8>/<document-id>.json; write-once"
        ),
        "resume_rule": (
            "validate and skip an existing exact-fingerprint response; retry missing ids"
        ),
        "failure_rule": (
            "record every failure and refuse matrix completion while any pair is missing"
        ),
        "canonicalization": {
            "unicode_policy": "nfc",
            "source": "raw responses only",
            "ground_truth_allowed": False,
        },
    }
    _write_new_json(PLAN, payload)
    return validate_plan()


def validate_plan() -> dict[str, Any]:
    plan = _read(PLAN)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise DevelopmentOcrError("unsupported development OCR plan schema")
    manifest, selected = _locked_selection()
    expected_bindings = {
        "role_manifest_sha256": file_sha256(ROLE_MANIFEST),
        "reserve_lock_sha256": file_sha256(LOCK),
        "pre_access_snapshot_sha256": file_sha256(SNAPSHOT),
        "c1_certificate_sha256": file_sha256(C1),
        "engine_config_sha256": file_sha256(ENGINE_CONFIG),
    }
    mismatches = {
        key: {"recorded": plan.get(key), "observed": value}
        for key, value in expected_bindings.items()
        if plan.get(key) != value
    }
    if mismatches:
        raise DevelopmentOcrError(f"development OCR plan input hash mismatch: {mismatches}")
    current_runner = file_sha256(Path(__file__).resolve())
    if plan.get("runner_sha256") != current_runner:
        if not PLAN_AMENDMENT.is_file():
            raise DevelopmentOcrError(
                "development OCR runner changed after plan freeze without a recorded amendment"
            )
        amendment = _read(PLAN_AMENDMENT)
        expected_amendment = {
            "schema_version": "sgv1-development-ocr-plan-amendment-v1",
            "base_plan_sha256": file_sha256(PLAN),
            "previous_runner_sha256": plan.get("runner_sha256"),
            "amended_runner_sha256": current_runner,
            "incident_sha256": file_sha256(INCIDENT),
        }
        amendment_mismatches = {
            key: {"recorded": amendment.get(key), "observed": value}
            for key, value in expected_amendment.items()
            if amendment.get(key) != value
        }
        if amendment_mismatches:
            raise DevelopmentOcrError(
                f"development OCR plan amendment mismatch: {amendment_mismatches}"
            )
    if plan.get("document_count") != len(selected):
        raise DevelopmentOcrError("development OCR document count changed after plan freeze")
    if plan.get("document_set_sha256") != stable_string_set_hash(selected):
        raise DevelopmentOcrError("development OCR selected document set changed after plan freeze")
    if plan.get("allowed_roles") != list(ALLOWED_ROLES):
        raise DevelopmentOcrError("development OCR allowed-role set drifted")
    if plan.get("engine_order") != list(ENGINE_ORDER):
        raise DevelopmentOcrError("development OCR engine set/order drifted")
    if manifest["counts"]["CONFIRMATORY"] != 177:
        raise DevelopmentOcrError("confirmatory reserve count changed after OCR plan freeze")
    return plan


def issue_validator_amendment() -> dict[str, Any]:
    """Record the stopped Tesseract checksum-auditor defect before resuming OCR."""
    if PLAN_AMENDMENT.exists() or INCIDENT.exists():
        raise DevelopmentOcrError("validator amendment/incident already exists; refusing rewrite")
    plan = _read(PLAN)
    current_runner = file_sha256(Path(__file__).resolve())
    if current_runner == plan.get("runner_sha256"):
        raise DevelopmentOcrError("runner has not changed; no validator amendment is needed")
    run_path = RUNS / "tesseract.json"
    if not run_path.is_file():
        raise DevelopmentOcrError("no interrupted Tesseract run record exists")
    interrupted = _read(run_path)
    if interrupted.get("status") != "INTERRUPTED":
        raise DevelopmentOcrError(
            f"expected INTERRUPTED Tesseract record, got {interrupted.get('status')}"
        )
    if not set(interrupted.get("failures", {})):
        raise DevelopmentOcrError("interrupted record has no validator failures to explain")

    _, selected = _locked_selection()
    rows = _pool_rows(selected)
    store = RawStore()
    fingerprint = str(plan["engines"]["tesseract"]["expected_fingerprint"])
    present: list[str] = []
    for document_id in selected:
        path = _raw_path(store, "tesseract", fingerprint, document_id)
        if not path.is_file():
            continue
        _validate_raw(
            path,
            store=store,
            document_id=document_id,
            engine_id="tesseract",
            fingerprint=fingerprint,
            image_sha256=str(rows[document_id]["image_sha256"]),
        )
        present.append(document_id)
    manifest = load_role_manifest(ROLE_MANIFEST)
    reserve_ids = [
        document_id for document_id, role in manifest["role_of"].items() if role == "CONFIRMATORY"
    ]
    reserve_responses = [
        document_id
        for document_id in reserve_ids
        if _raw_path(store, "tesseract", fingerprint, document_id).is_file()
    ]
    if reserve_responses:
        raise DevelopmentOcrError(
            f"validator incident audit found {len(reserve_responses)} reserve response(s)"
        )
    if len(present) != interrupted.get("raw_responses_present"):
        raise DevelopmentOcrError(
            "interrupted run census disagrees with independently validated raw responses"
        )

    incident = {
        "schema_version": "sgv1-development-ocr-incident-v1",
        "issued_utc": _utc_now(),
        "classification": "audit-wrapper defect; raw OCR responses valid",
        "original_run_record_path": run_path.relative_to(REPO).as_posix(),
        "original_run_record_sha256": file_sha256(run_path),
        "original_run_record": interrupted,
        "root_cause": (
            "The wrapper recomputed every engine's payload checksum over canonical JSON. "
            "The Tesseract adapter contract instead hashes verbatim TSV bytes."
        ),
        "responses_written_before_stop": len(present),
        "responses_valid_under_adapter_contract": len(present),
        "validated_response_ids_sha256": stable_string_set_hash(present),
        "confirmatory_responses_present": reserve_responses,
        "raw_responses_deleted_or_rewritten": False,
        "role_or_document_selection_changed": False,
        "engine_or_fingerprint_changed": False,
        "scientific_data_impact": "none; recognition bytes were valid and remain write-once",
    }
    _write_new_json(INCIDENT, incident)
    amendment = {
        "schema_version": "sgv1-development-ocr-plan-amendment-v1",
        "issued_utc": _utc_now(),
        "base_plan_path": PLAN.relative_to(REPO).as_posix(),
        "base_plan_sha256": file_sha256(PLAN),
        "previous_runner_sha256": plan["runner_sha256"],
        "amended_runner_sha256": current_runner,
        "incident_path": INCIDENT.relative_to(REPO).as_posix(),
        "incident_sha256": file_sha256(INCIDENT),
        "change_scope": "raw-response checksum validation only",
        "change": (
            "Validate tesseract_tsv_v5 payload_sha256 against UTF-8 TSV bytes; retain "
            "canonical-JSON payload validation for PaddleOCR, EasyOCR, and docTR."
        ),
        "unchanged_invariants": {
            "role_manifest_sha256": plan["role_manifest_sha256"],
            "reserve_lock_sha256": plan["reserve_lock_sha256"],
            "document_count": plan["document_count"],
            "document_set_sha256": plan["document_set_sha256"],
            "engine_order": plan["engine_order"],
            "expected_fingerprints": {
                engine_id: plan["engines"][engine_id]["expected_fingerprint"]
                for engine_id in ENGINE_ORDER
            },
        },
        "resume_disposition": (
            "reuse the 82 validated immutable responses and continue missing ids"
        ),
    }
    _write_new_json(PLAN_AMENDMENT, amendment)
    validate_plan()
    return amendment


def _adapter(plan: dict[str, Any], engine_id: str) -> tuple[Any, dict[str, Any]]:
    if engine_id not in ENGINE_ORDER:
        raise DevelopmentOcrError(f"unknown or unplanned OCR engine {engine_id}")
    spec: dict[str, Any] = plan["engines"][engine_id]
    adapter = build_engine(spec["adapter"], engine_id=engine_id, **spec["params"])
    availability = adapter.availability()
    availability.require()
    fingerprint = adapter.fingerprint()
    if fingerprint.fingerprint != spec["expected_fingerprint"]:
        raise DevelopmentOcrError(
            f"{engine_id} fingerprint drift: {fingerprint.fingerprint} != "
            f"{spec['expected_fingerprint']}"
        )
    runtime = {
        "availability_version": availability.version,
        "engine_version": fingerprint.engine_version,
        "model_ids": list(fingerprint.model_ids),
        "config_sha256": fingerprint.config_hash,
        "fingerprint": fingerprint.fingerprint,
        "params": spec["params"],
        "device_policy": spec["device_policy"],
    }
    return adapter, runtime


def preflight(engine_id: str) -> dict[str, Any]:
    plan = validate_plan()
    _, runtime = _adapter(plan, engine_id)
    _, selected = _locked_selection()
    store = RawStore()
    existing = sum(
        _raw_path(store, engine_id, runtime["fingerprint"], document_id).is_file()
        for document_id in selected
    )
    print(
        f"{engine_id}: available version={runtime['engine_version']} "
        f"fingerprint={runtime['fingerprint'][:12]} existing={existing}/{len(selected)}"
    )
    return runtime


def run_engine(engine_id: str) -> int:
    plan = validate_plan()
    adapter, runtime = _adapter(plan, engine_id)
    _, selected = _locked_selection()
    rows = _pool_rows(selected)
    store = RawStore()
    record_path = RUNS / f"{engine_id}.json"
    previous = _read(record_path) if record_path.exists() else {}
    if previous and (
        previous.get("plan_sha256") != file_sha256(PLAN)
        or previous.get("document_set_sha256") != plan["document_set_sha256"]
        or previous.get("engine", {}).get("fingerprint") != runtime["fingerprint"]
    ):
        raise DevelopmentOcrError(f"existing {engine_id} run record binds different inputs")

    started_utc = str(previous.get("started_utc") or _utc_now())
    failures: dict[str, dict[str, str]] = dict(previous.get("failures", {}))
    process_started = time.monotonic()

    def checkpoint(status: str, completed: int, *, last_document_id: str | None) -> None:
        current_existing = sum(
            _raw_path(store, engine_id, runtime["fingerprint"], document_id).is_file()
            for document_id in selected
        )
        payload = {
            "schema_version": "sgv1-development-ocr-engine-run-v1",
            "status": status,
            "engine_id": engine_id,
            "engine": runtime,
            "plan_path": PLAN.relative_to(REPO).as_posix(),
            "plan_sha256": file_sha256(PLAN),
            "plan_amendment_sha256": (
                file_sha256(PLAN_AMENDMENT) if PLAN_AMENDMENT.is_file() else None
            ),
            "role_manifest_sha256": file_sha256(ROLE_MANIFEST),
            "document_count": len(selected),
            "document_set_sha256": plan["document_set_sha256"],
            "raw_responses_present": current_existing,
            "missing_responses": len(selected) - current_existing,
            "completed_in_this_invocation": completed,
            "failures": failures,
            "started_utc": started_utc,
            "last_checkpoint_utc": _utc_now(),
            "last_document_id": last_document_id,
            "pid": os.getpid(),
            "command": [sys.executable, *sys.argv],
            "git": {
                "branch": _git_value("branch", "--show-current"),
                "head": _git_value("rev-parse", "HEAD"),
            },
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "invocation_elapsed_seconds": time.monotonic() - process_started,
            "peak_memory_bytes": _peak_memory_bytes(),
        }
        _write_json_atomic(record_path, payload)

    completed = 0
    cached = 0
    checkpoint("RUNNING", completed, last_document_id=None)
    try:
        for index, document_id in enumerate(selected, start=1):
            row = rows[document_id]
            raw_path = _raw_path(store, engine_id, runtime["fingerprint"], document_id)
            if raw_path.is_file():
                _validate_raw(
                    raw_path,
                    store=store,
                    document_id=document_id,
                    engine_id=engine_id,
                    fingerprint=runtime["fingerprint"],
                    image_sha256=str(row["image_sha256"]),
                )
                failures.pop(document_id, None)
                cached += 1
                continue

            # Only now, after the whole id selection passed the reserve guard, is a source
            # path resolved and read. The byte hash is checked before the OCR call.
            image_path = data_root() / str(row["image_path"])
            if not image_path.is_file():
                failures[document_id] = {
                    "type": "FileNotFoundError",
                    "message": f"missing source image {image_path.relative_to(REPO)}",
                }
                checkpoint("RUNNING_WITH_FAILURES", completed, last_document_id=document_id)
                continue
            if file_sha256(image_path) != row["image_sha256"]:
                failures[document_id] = {
                    "type": "SourceChecksumMismatch",
                    "message": "source image differs from the C1 identity manifest",
                }
                checkpoint("RUNNING_WITH_FAILURES", completed, last_document_id=document_id)
                continue
            page = PageInput(
                document_id=document_id,
                dataset_id="cord",
                image_path=image_path,
                image_sha256=str(row["image_sha256"]),
                width=int(row["width"]),
                height=int(row["height"]),
            )
            try:
                response = adapter.recognize(page)
                store.write_engine_response(response)
                _validate_raw(
                    raw_path,
                    store=store,
                    document_id=document_id,
                    engine_id=engine_id,
                    fingerprint=runtime["fingerprint"],
                    image_sha256=str(row["image_sha256"]),
                )
            except Exception as error:  # record and continue; never silently drop a page
                failures[document_id] = {
                    "type": type(error).__name__,
                    "message": str(error)[:500],
                }
            else:
                failures.pop(document_id, None)
                completed += 1
            if index % 10 == 0 or failures.get(document_id):
                checkpoint(
                    "RUNNING_WITH_FAILURES" if failures else "RUNNING",
                    completed,
                    last_document_id=document_id,
                )
            if index % 25 == 0:
                print(
                    f"{engine_id}: {index}/{len(selected)} visited, "
                    f"new={completed} cached={cached} failures={len(failures)}",
                    flush=True,
                )
    except KeyboardInterrupt:
        checkpoint("INTERRUPTED", completed, last_document_id=None)
        raise

    present = sum(
        _raw_path(store, engine_id, runtime["fingerprint"], document_id).is_file()
        for document_id in selected
    )
    status = "COMPLETE" if present == len(selected) and not failures else "INCOMPLETE"
    checkpoint(status, completed, last_document_id=selected[-1])
    print(
        f"{engine_id}: status={status} present={present}/{len(selected)} "
        f"new={completed} cached={cached} failures={len(failures)}",
        flush=True,
    )
    return 0 if status == "COMPLETE" else 2


def canonicalize_and_freeze() -> int:
    plan = validate_plan()
    _, selected = _locked_selection()
    rows = _pool_rows(selected)
    store = RawStore()
    manifest_rows: list[dict[str, Any]] = []
    canonical_rows: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()
    missing: list[dict[str, str]] = []

    if RAW_MANIFEST.exists() or CANONICAL_SPANS.exists() or CANONICAL_FREEZE.exists():
        raise DevelopmentOcrError(
            "canonical OCR freeze artifacts already exist; audit them instead of rewriting"
        )
    for engine_id in ENGINE_ORDER:
        run_path = RUNS / f"{engine_id}.json"
        if not run_path.is_file():
            raise DevelopmentOcrError(f"missing completed OCR run record for {engine_id}")
        run = _read(run_path)
        if run.get("status") != "COMPLETE" or run.get("failures"):
            raise DevelopmentOcrError(
                f"{engine_id} OCR run is not cleanly complete: "
                f"status={run.get('status')} failures={len(run.get('failures', {}))}"
            )
        if run.get("plan_sha256") != file_sha256(PLAN):
            raise DevelopmentOcrError(f"{engine_id} OCR run binds a different plan")

    policy = CanonicalizationPolicy(unicode_policy="nfc")
    for engine_id in ENGINE_ORDER:
        spec: dict[str, Any] = plan["engines"][engine_id]
        adapter = build_engine(spec["adapter"], engine_id=engine_id, **spec["params"])
        fingerprint = str(spec["expected_fingerprint"])
        for document_id in selected:
            raw_path = _raw_path(store, engine_id, fingerprint, document_id)
            if not raw_path.is_file():
                missing.append({"document_id": document_id, "engine_id": engine_id})
                continue
            raw_meta = _validate_raw(
                raw_path,
                store=store,
                document_id=document_id,
                engine_id=engine_id,
                fingerprint=fingerprint,
                image_sha256=str(rows[document_id]["image_sha256"]),
            )
            raw = store.read_engine_response(raw_path)
            spans = canonicalize_response(
                raw=raw,
                parsed=adapter.parse(raw),
                policy=policy,
                conf_scale_name=adapter.confidence_scale.name,
                raw_ref=raw_path.relative_to(data_root()).as_posix(),
            )
            pair_counts[engine_id] += len(spans)
            canonical_rows.extend(span.model_dump(mode="json") for span in spans)
            manifest_rows.append(
                {
                    "document_id": document_id,
                    "engine_id": engine_id,
                    "engine_fingerprint": fingerprint,
                    "image_sha256": rows[document_id]["image_sha256"],
                    "raw_path": raw_path.relative_to(REPO).as_posix(),
                    **raw_meta,
                    "canonical_span_count": len(spans),
                }
            )
    if missing:
        raise DevelopmentOcrError(
            f"OCR matrix is missing {len(missing)} document-engine response(s); freeze refused"
        )
    if len(manifest_rows) != len(selected) * len(ENGINE_ORDER):
        raise DevelopmentOcrError("OCR matrix census does not equal documents x engines")
    forbidden = {"gt_text", "ground_truth", "outcome", "d_before", "d_after", "label"}
    present_columns = set(canonical_rows[0]) if canonical_rows else set()
    leaking = sorted(present_columns & forbidden)
    if leaking:
        raise DevelopmentOcrError(f"canonical OCR rows contain GT/label columns {leaking}")

    _write_new_json(
        RAW_MANIFEST,
        {
            "schema_version": "sgv1-development-raw-ocr-manifest-v1",
            "issued_utc": _utc_now(),
            "plan_sha256": file_sha256(PLAN),
            "document_count": len(selected),
            "engine_count": len(ENGINE_ORDER),
            "pair_count": len(manifest_rows),
            "document_set_sha256": plan["document_set_sha256"],
            "records": manifest_rows,
        },
    )
    table = pa.Table.from_pylist(canonical_rows)
    CANONICAL_SPANS.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, CANONICAL_SPANS, compression="zstd")
    zero_span = [
        {"document_id": row["document_id"], "engine_id": row["engine_id"]}
        for row in manifest_rows
        if row["canonical_span_count"] == 0
    ]
    durations = {
        engine_id: sum(
            float(row["duration_seconds"]) for row in manifest_rows if row["engine_id"] == engine_id
        )
        for engine_id in ENGINE_ORDER
    }
    freeze = {
        "schema_version": "sgv1-development-canonical-ocr-freeze-v1",
        "issued_utc": _utc_now(),
        "frozen": True,
        "scientific_scope": "TRAIN + CALIBRATION + DEVELOPMENT only",
        "confirmatory_accessed": False,
        "plan_path": PLAN.relative_to(REPO).as_posix(),
        "plan_sha256": file_sha256(PLAN),
        "plan_amendment_sha256": (
            file_sha256(PLAN_AMENDMENT) if PLAN_AMENDMENT.is_file() else None
        ),
        "role_manifest_sha256": file_sha256(ROLE_MANIFEST),
        "reserve_lock_sha256": file_sha256(LOCK),
        "document_count": len(selected),
        "engine_count": len(ENGINE_ORDER),
        "expected_pair_count": len(selected) * len(ENGINE_ORDER),
        "actual_pair_count": len(manifest_rows),
        "missing_pairs": missing,
        "zero_span_pairs": zero_span,
        "canonical_span_count": len(canonical_rows),
        "canonical_spans_by_engine": dict(pair_counts),
        "raw_manifest_path": RAW_MANIFEST.relative_to(REPO).as_posix(),
        "raw_manifest_sha256": file_sha256(RAW_MANIFEST),
        "canonical_spans_path": CANONICAL_SPANS.relative_to(REPO).as_posix(),
        "canonical_spans_sha256": file_sha256(CANONICAL_SPANS),
        "canonical_columns": sorted(present_columns),
        "gt_or_label_columns": leaking,
        "canonicalization_policy": {"unicode_policy": "nfc"},
        "engine_fingerprints": {
            engine_id: plan["engines"][engine_id]["expected_fingerprint"]
            for engine_id in ENGINE_ORDER
        },
        "recognition_duration_seconds_by_engine": durations,
        "engine_run_records": {
            engine_id: {
                "path": (RUNS / f"{engine_id}.json").relative_to(REPO).as_posix(),
                "sha256": file_sha256(RUNS / f"{engine_id}.json"),
                "status": _read(RUNS / f"{engine_id}.json")["status"],
            }
            for engine_id in ENGINE_ORDER
        },
    }
    _write_new_json(CANONICAL_FREEZE, freeze)
    print(
        f"canonical OCR frozen: {len(canonical_rows)} spans, "
        f"{len(manifest_rows)} document-engine pairs, zero_span={len(zero_span)}"
    )
    print(f"canonical_sha256: {freeze['canonical_spans_sha256']}")
    return 0


def audit_existing() -> int:
    plan = validate_plan()
    _, selected = _locked_selection()
    rows = _pool_rows(selected)
    store = RawStore()
    counts: dict[str, int] = {}
    failures: list[str] = []
    for engine_id in ENGINE_ORDER:
        fingerprint = str(plan["engines"][engine_id]["expected_fingerprint"])
        count = 0
        for document_id in selected:
            path = _raw_path(store, engine_id, fingerprint, document_id)
            if not path.is_file():
                failures.append(f"missing:{engine_id}:{document_id}")
                continue
            try:
                _validate_raw(
                    path,
                    store=store,
                    document_id=document_id,
                    engine_id=engine_id,
                    fingerprint=fingerprint,
                    image_sha256=str(rows[document_id]["image_sha256"]),
                )
            except DevelopmentOcrError as error:
                failures.append(str(error))
            else:
                count += 1
        counts[engine_id] = count
    print("OCR matrix: " + ", ".join(f"{engine}={counts[engine]}" for engine in ENGINE_ORDER))
    if failures:
        print(f"audit failures: {len(failures)}")
        for failure in failures[:20]:
            print(f"  {failure}")
        return 2
    if CANONICAL_FREEZE.exists():
        freeze = _read(CANONICAL_FREEZE)
        if file_sha256(CANONICAL_SPANS) != freeze["canonical_spans_sha256"]:
            raise DevelopmentOcrError("canonical span parquet changed after freeze")
        if file_sha256(RAW_MANIFEST) != freeze["raw_manifest_sha256"]:
            raise DevelopmentOcrError("raw response manifest changed after freeze")
        print(f"canonical freeze: PASS ({freeze['canonical_span_count']} spans)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze-plan", action="store_true")
    actions.add_argument("--issue-validator-amendment", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--canonicalize", action="store_true")
    actions.add_argument("--audit", action="store_true")
    parser.add_argument("--engine", choices=ENGINE_ORDER)
    args = parser.parse_args()
    if args.freeze_plan:
        plan = freeze_plan()
        print(
            f"development OCR plan frozen: {plan['document_count']} documents x "
            f"{len(plan['engine_order'])} engines"
        )
        print(f"document_set_sha256: {plan['document_set_sha256']}")
        return 0
    if args.issue_validator_amendment:
        amendment = issue_validator_amendment()
        print("validator amendment issued: Tesseract TSV checksum contract")
        print(f"amended_runner_sha256: {amendment['amended_runner_sha256']}")
        return 0
    if args.preflight:
        if not args.engine:
            parser.error("--preflight requires --engine")
        preflight(args.engine)
        return 0
    if args.run:
        if not args.engine:
            parser.error("--run requires --engine")
        return run_engine(args.engine)
    if args.canonicalize:
        if args.engine:
            parser.error("--canonicalize does not accept --engine")
        return canonicalize_and_freeze()
    return audit_existing()


if __name__ == "__main__":
    raise SystemExit(main())
