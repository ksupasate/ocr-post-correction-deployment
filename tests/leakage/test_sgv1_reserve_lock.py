"""Adversarial tests for the SGV1 role split and untouched reserve boundary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ocr_risk.experiments.sgv1_reserve import (
    CALIBRATION,
    CONFIRMATORY,
    DEVELOPMENT,
    TRAIN,
    ReserveAccessError,
    RoleManifestError,
    assert_sgv1_development_access,
    build_role_assignment,
    load_reserve_lock,
    load_role_manifest,
    role_assignment_digest,
    validate_role_manifest,
)
from ocr_risk.io.hashing import stable_string_set_hash

REPO = Path(__file__).resolve().parents[2]
ROLE_MANIFEST = REPO / "manifests/sgv1/role_manifest.json"
LOCK = REPO / "manifests/sgv1/confirmatory_reserve_lock.json"
SNAPSHOT = REPO / "results/generated/sgv1/reserve/pre_access_freshness_snapshot.json"
RED_TEAM = REPO / "results/generated/sgv1/reserve/reserve_lock_red_team.json"


def test_role_assignment_is_deterministic_duplicate_atomic_and_preaccess_safe() -> None:
    ids = [f"d{index}" for index in range(20)]
    groups = [["d0", "d1"], ["d7", "d8"]]
    targets = {TRAIN: 10, CALIBRATION: 3, DEVELOPMENT: 3, CONFIRMATORY: 4}
    first, first_audit = build_role_assignment(
        ids,
        groups,
        ["d0"],
        seed="unit-test",
        target_counts=targets,
    )
    second, second_audit = build_role_assignment(
        ids,
        groups,
        ["d0"],
        seed="unit-test",
        target_counts=targets,
    )
    assert first == second
    assert first_audit == second_audit
    assert first["d0"] == first["d1"] != CONFIRMATORY
    assert first["d7"] == first["d8"]
    assert {role: list(first.values()).count(role) for role in targets} == targets


def test_atomic_assignment_fails_when_exact_target_is_impossible() -> None:
    with pytest.raises(RoleManifestError, match="without splitting a duplicate"):
        build_role_assignment(
            ["d0", "d1", "d2", "d3"],
            [["d0", "d1"], ["d2", "d3"]],
            [],
            seed="unit-test",
            target_counts={TRAIN: 1, CALIBRATION: 1, DEVELOPMENT: 1, CONFIRMATORY: 1},
        )


def test_frozen_role_manifest_and_lock_rederive() -> None:
    manifest = load_role_manifest(ROLE_MANIFEST)
    lock = load_reserve_lock(
        LOCK,
        role_manifest_path=ROLE_MANIFEST,
        snapshot_path=SNAPSHOT,
    )
    assert manifest["c1"]["status"] == "C1_PASS"
    assert manifest["counts"] == {
        TRAIN: 444,
        CALIBRATION: 133,
        DEVELOPMENT: 133,
        CONFIRMATORY: 177,
    }
    assert lock["status"] == "LOCKED"
    assert lock["reserve_count"] == 177
    assert len(manifest["role_atomic_groups"]) == 57
    reserve = {
        document_id for document_id, role in manifest["role_of"].items() if role == CONFIRMATORY
    }
    assert reserve.isdisjoint(manifest["reserve_constraints"]["barred_ids"])


def test_preaccess_snapshot_and_required_red_team_are_clean() -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    red_team = json.loads(RED_TEAM.read_text(encoding="utf-8"))
    assert snapshot["clean"] is True
    assert snapshot["downstream_access_hits"] == []
    assert snapshot["reserve_count"] == 177
    assert red_team["all_required_violations_blocked"] is True
    assert len(red_team["attempts"]) == 4
    assert all(attempt["expected_failure_observed"] for attempt in red_team["attempts"])


@pytest.mark.parametrize(
    "operation",
    ["development OCR", "candidate annotation selection", "Track-A image load"],
)
def test_confirmatory_id_is_blocked_from_every_development_surface(operation: str) -> None:
    manifest = load_role_manifest(ROLE_MANIFEST)
    reserve_id = next(
        document_id for document_id, role in manifest["role_of"].items() if role == CONFIRMATORY
    )
    with pytest.raises(ReserveAccessError, match="CONFIRMATORY reserve"):
        assert_sgv1_development_access(
            reserve_id,
            operation=operation,
            role_manifest_path=ROLE_MANIFEST,
            lock_path=LOCK,
            snapshot_path=SNAPSHOT,
        )


def test_nonconfirmatory_ids_pass_the_locked_guard() -> None:
    manifest = load_role_manifest(ROLE_MANIFEST)
    selected = [
        next(document_id for document_id, value in manifest["role_of"].items() if value == role)
        for role in (TRAIN, CALIBRATION, DEVELOPMENT)
    ]
    assert_sgv1_development_access(
        selected,
        operation="unit-test development selection",
        role_manifest_path=ROLE_MANIFEST,
        lock_path=LOCK,
        snapshot_path=SNAPSHOT,
    )


def test_unknown_document_id_fails_closed() -> None:
    with pytest.raises(ReserveAccessError, match="absent from the frozen"):
        assert_sgv1_development_access(
            "cord-train-9999",
            operation="development OCR",
            role_manifest_path=ROLE_MANIFEST,
            lock_path=LOCK,
            snapshot_path=SNAPSHOT,
        )


def test_duplicate_sibling_cannot_be_moved_to_another_role_even_if_rehashed() -> None:
    manifest = load_role_manifest(ROLE_MANIFEST)
    tampered = copy.deepcopy(manifest)
    left, right = tampered["role_atomic_groups"][0]
    current = tampered["role_of"][left]
    replacement = next(role for role in (TRAIN, CALIBRATION, DEVELOPMENT) if role != current)
    tampered["role_of"][right] = replacement
    tampered["counts"] = {
        role: list(tampered["role_of"].values()).count(role)
        for role in (TRAIN, CALIBRATION, DEVELOPMENT, CONFIRMATORY)
    }
    tampered["document_set_sha256"] = {
        role: stable_string_set_hash(
            document_id for document_id, value in tampered["role_of"].items() if value == role
        )
        for role in (TRAIN, CALIBRATION, DEVELOPMENT, CONFIRMATORY)
    }
    tampered["assignment_sha256"] = role_assignment_digest(
        tampered["split_rule"]["spec_sha256"], tampered["role_of"]
    )
    with pytest.raises(RoleManifestError, match=r"duplicate group.*crosses"):
        validate_role_manifest(tampered)
