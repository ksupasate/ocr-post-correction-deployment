"""SGV1 four-way role assignment and fail-closed confirmatory-reserve guard.

The CORD role manifest is drawn before OCR outcomes exist.  Assignment uses document
identifiers and the C1 duplicate graph only: no image, annotation, OCR, candidate, or
verifier value is an input.  Confirmatory membership is then treated as a hard access
boundary by every SGV1 development entry point.

Mechanical corpus-identity hashing during C1 is not scientific consumption.  Once this
module issues the reserve lock, however, even an attempted development access by id is a
runtime error.  The guard deliberately knows ids and hashes, not document content.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from ocr_risk.io.hashing import canonical_hash, file_sha256, stable_string_set_hash

__all__ = [
    "CALIBRATION",
    "CONFIRMATORY",
    "DEVELOPMENT",
    "DEVELOPMENT_ROLES",
    "ROLE_ORDER",
    "TRAIN",
    "ReserveAccessError",
    "RoleManifestError",
    "assert_duplicate_group_integrity",
    "assert_sgv1_development_access",
    "build_role_assignment",
    "confirmatory_document_ids",
    "load_reserve_lock",
    "load_role_manifest",
    "role_assignment_digest",
    "validate_role_manifest",
]

TRAIN: Final = "TRAIN"
CALIBRATION: Final = "CALIBRATION"
DEVELOPMENT: Final = "DEVELOPMENT"
CONFIRMATORY: Final = "CONFIRMATORY"
ROLE_ORDER: Final = (TRAIN, CALIBRATION, DEVELOPMENT, CONFIRMATORY)
DEVELOPMENT_ROLES: Final = frozenset({TRAIN, CALIBRATION, DEVELOPMENT})


class RoleManifestError(RuntimeError):
    """A role or reserve artifact is incomplete, inconsistent, or tampered with."""


class ReserveAccessError(RoleManifestError):
    """A confirmatory document reached an SGV1 development operation."""


def _role_units(
    document_ids: Sequence[str], duplicate_groups: Sequence[Sequence[str]]
) -> tuple[tuple[str, ...], ...]:
    """Return connected duplicate components plus singleton documents.

    C1 currently emits disjoint components, but rebuilding the connected components here
    makes overlapping future detector outputs safe instead of silently splitting a page.
    """
    ids = set(document_ids)
    parent = {document_id: document_id for document_id in ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for raw_group in duplicate_groups:
        group = sorted(set(raw_group) & ids)
        for member in group[1:]:
            union(group[0], member)

    members: dict[str, list[str]] = {}
    for document_id in sorted(ids):
        members.setdefault(find(document_id), []).append(document_id)
    return tuple(sorted((tuple(group) for group in members.values()), key=lambda group: group[0]))


def _ordered(units: Iterable[tuple[str, ...]], *, seed: str, stage: str) -> list[tuple[str, ...]]:
    return sorted(
        units,
        key=lambda unit: (
            canonical_hash({"seed": seed, "stage": stage, "role_unit": list(unit)}),
            unit,
        ),
    )


def _take_document_target(
    units: Sequence[tuple[str, ...]], target: int, *, seed: str, stage: str
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """Take whole units up to an exact document target, deterministically.

    A size-two duplicate group is skipped temporarily when only one slot remains.  With
    the CORD pool's many singleton units this reaches every frozen target exactly; if a
    future pool cannot, assignment fails rather than moving a duplicate across roles or
    silently changing the requested proportions.
    """
    selected: list[tuple[str, ...]] = []
    remainder: list[tuple[str, ...]] = []
    selected_documents = 0
    for unit in _ordered(units, seed=seed, stage=stage):
        if selected_documents + len(unit) <= target:
            selected.append(unit)
            selected_documents += len(unit)
        else:
            remainder.append(unit)
    if selected_documents != target:
        raise RoleManifestError(
            f"cannot assign exact {stage} target {target} without splitting a duplicate "
            f"group; atomic allocation reached {selected_documents}"
        )
    return selected, remainder


def role_assignment_digest(spec_sha256: str, role_of: Mapping[str, str]) -> str:
    """Hash the frozen split rule and exact assignment together."""
    return canonical_hash(
        {
            "spec_sha256": spec_sha256,
            "role_of": sorted(
                (str(document_id), str(role)) for document_id, role in role_of.items()
            ),
        }
    )


def build_role_assignment(
    document_ids: Sequence[str],
    duplicate_groups: Sequence[Sequence[str]],
    preaccess_development_only: Iterable[str],
    *,
    seed: str,
    target_counts: Mapping[str, int],
) -> tuple[dict[str, str], dict[str, object]]:
    """Assign C1-admissible documents without observing content or downstream outcomes."""
    ids = sorted(set(document_ids))
    if len(ids) != len(document_ids):
        raise RoleManifestError("admissible document ids are not unique")
    if set(target_counts) != set(ROLE_ORDER):
        raise RoleManifestError(f"target counts must name exactly {ROLE_ORDER}")
    if sum(target_counts.values()) != len(ids):
        raise RoleManifestError(
            f"target counts sum to {sum(target_counts.values())}, not the {len(ids)} documents"
        )
    if any(count < 1 for count in target_counts.values()):
        raise RoleManifestError("every SGV1 role must contain at least one document")

    units = _role_units(ids, duplicate_groups)
    preaccess = set(preaccess_development_only)
    unknown_preaccess = preaccess - set(ids)
    if unknown_preaccess:
        raise RoleManifestError(
            f"pre-access exclusions include {len(unknown_preaccess)} non-admissible ids"
        )
    barred_units = [unit for unit in units if set(unit) & preaccess]
    confirmatory_units = [unit for unit in units if not set(unit) & preaccess]
    selected_confirmatory, unused_eligible = _take_document_target(
        confirmatory_units,
        target_counts[CONFIRMATORY],
        seed=seed,
        stage="confirmatory",
    )
    remaining_units = [*unused_eligible, *barred_units]
    selected_train, remaining_units = _take_document_target(
        remaining_units,
        target_counts[TRAIN],
        seed=seed,
        stage="train",
    )
    selected_calibration, selected_development = _take_document_target(
        remaining_units,
        target_counts[CALIBRATION],
        seed=seed,
        stage="calibration",
    )
    development_count = sum(len(unit) for unit in selected_development)
    if development_count != target_counts[DEVELOPMENT]:
        raise RoleManifestError(
            f"development remainder has {development_count} documents, expected "
            f"{target_counts[DEVELOPMENT]}"
        )

    units_by_role = {
        TRAIN: selected_train,
        CALIBRATION: selected_calibration,
        DEVELOPMENT: selected_development,
        CONFIRMATORY: selected_confirmatory,
    }
    role_of = {
        document_id: role
        for role in ROLE_ORDER
        for unit in units_by_role[role]
        for document_id in unit
    }
    if set(role_of) != set(ids):
        raise RoleManifestError("role assignment does not cover the admissible pool exactly")
    assert_duplicate_group_integrity(role_of, duplicate_groups)

    propagated = sorted(
        {
            member
            for unit in barred_units
            if set(unit) & preaccess
            for member in unit
            if member not in preaccess
        }
    )
    allocation = {
        "n_role_units": len(units),
        "n_duplicate_groups": sum(len(unit) > 1 for unit in units),
        "preaccess_development_only_ids": sorted(preaccess),
        "preaccess_duplicate_propagated_ids": propagated,
        "confirmatory_eligible_after_duplicate_propagation": sum(
            len(unit) for unit in confirmatory_units
        ),
    }
    return role_of, allocation


def assert_duplicate_group_integrity(
    role_of: Mapping[str, str], duplicate_groups: Sequence[Sequence[str]]
) -> None:
    """Reject an assignment that puts any exact/near duplicate component in two roles."""
    for raw_group in duplicate_groups:
        group = sorted({str(item) for item in raw_group} & set(role_of))
        roles = {role_of[member] for member in group}
        if len(roles) > 1:
            raise RoleManifestError(
                f"duplicate group {group} crosses SGV1 roles {sorted(roles)}; "
                "document-level leakage is forbidden"
            )


def validate_role_manifest(payload: Mapping[str, Any]) -> None:
    """Re-derive all self-contained role-manifest invariants."""
    if payload.get("schema_version") != "sgv1-role-manifest-v1":
        raise RoleManifestError("unsupported SGV1 role-manifest schema")
    raw_role_of = payload.get("role_of")
    if not isinstance(raw_role_of, dict):
        raise RoleManifestError("role manifest has no role_of mapping")
    role_of = {str(document_id): str(role) for document_id, role in raw_role_of.items()}
    invalid_roles = set(role_of.values()) - set(ROLE_ORDER)
    if invalid_roles:
        raise RoleManifestError(f"role manifest contains invalid roles {sorted(invalid_roles)}")

    spec_sha256 = str(payload.get("split_rule", {}).get("spec_sha256", ""))
    observed_assignment = role_assignment_digest(spec_sha256, role_of)
    if observed_assignment != payload.get("assignment_sha256"):
        raise RoleManifestError("role assignment hash does not match the manifest contents")

    counts = {role: sum(value == role for value in role_of.values()) for role in ROLE_ORDER}
    if counts != payload.get("counts"):
        raise RoleManifestError(f"role counts do not rederive: {counts} != {payload.get('counts')}")
    digests = {
        role: stable_string_set_hash(
            document_id for document_id, value in role_of.items() if value == role
        )
        for role in ROLE_ORDER
    }
    if digests != payload.get("document_set_sha256"):
        raise RoleManifestError("role document-set hashes do not rederive")

    raw_groups = payload.get("role_atomic_groups")
    if not isinstance(raw_groups, list):
        raise RoleManifestError("role manifest has no role_atomic_groups list")
    groups = [[str(item) for item in group] for group in raw_groups]
    assert_duplicate_group_integrity(role_of, groups)

    barred = {str(item) for item in payload.get("reserve_constraints", {}).get("barred_ids", [])}
    offenders = sorted(
        document_id for document_id in barred if role_of.get(document_id) == CONFIRMATORY
    )
    if offenders:
        raise RoleManifestError(
            f"{len(offenders)} pre-access/duplicate-constrained documents entered CONFIRMATORY"
        )
    if counts[CONFIRMATORY] < 1:
        raise RoleManifestError("confirmatory reserve is empty")


def load_role_manifest(path: Path | str) -> dict[str, Any]:
    """Load and validate the frozen role manifest."""
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_role_manifest(payload)
    return payload


def confirmatory_document_ids(path: Path | str) -> frozenset[str]:
    """Return confirmatory ids without loading any document content."""
    payload = load_role_manifest(path)
    return frozenset(
        document_id for document_id, role in payload["role_of"].items() if role == CONFIRMATORY
    )


def load_reserve_lock(
    lock_path: Path | str,
    *,
    role_manifest_path: Path | str,
    snapshot_path: Path | str,
) -> dict[str, Any]:
    """Validate the reserve lock and every exact byte binding it cites."""
    lock: dict[str, Any] = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("schema_version") != "sgv1-confirmatory-reserve-lock-v1":
        raise RoleManifestError("unsupported SGV1 reserve-lock schema")
    if lock.get("status") != "LOCKED":
        raise RoleManifestError("SGV1 confirmatory reserve is not LOCKED")
    if file_sha256(role_manifest_path) != lock.get("role_manifest_sha256"):
        raise RoleManifestError("reserve lock does not bind the current role-manifest bytes")
    if file_sha256(snapshot_path) != lock.get("pre_access_snapshot_sha256"):
        raise RoleManifestError("reserve lock does not bind the current pre-access snapshot")

    manifest = load_role_manifest(role_manifest_path)
    reserve = sorted(
        document_id for document_id, role in manifest["role_of"].items() if role == CONFIRMATORY
    )
    snapshot: dict[str, Any] = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    if not snapshot.get("clean") or snapshot.get("downstream_access_hits"):
        raise RoleManifestError("pre-access snapshot is not clean")
    reserve_digest = stable_string_set_hash(reserve)
    if reserve_digest != lock.get("reserve_document_set_sha256"):
        raise RoleManifestError("reserve document-set hash does not rederive")
    if reserve_digest != snapshot.get("reserve_document_set_sha256"):
        raise RoleManifestError("snapshot and lock bind different reserve document sets")
    if len(reserve) != lock.get("reserve_count"):
        raise RoleManifestError("reserve count does not rederive")
    return lock


def assert_sgv1_development_access(
    document_ids: object,
    *,
    operation: str,
    role_manifest_path: Path | str,
    lock_path: Path | str,
    snapshot_path: Path | str,
) -> None:
    """Refuse reserve or unknown ids before any SGV1 development data access.

    Call this on identifiers *before* resolving image/annotation paths, constructing an
    OCR engine, loading Track-A pages, or selecting candidate rows.
    """
    load_reserve_lock(
        lock_path,
        role_manifest_path=role_manifest_path,
        snapshot_path=snapshot_path,
    )
    manifest = load_role_manifest(role_manifest_path)
    role_of: dict[str, str] = manifest["role_of"]
    selected: tuple[str, ...]
    if isinstance(document_ids, str):
        selected = (document_ids,)
    elif isinstance(document_ids, Iterable):
        selected = tuple(str(item) for item in document_ids)
    else:
        raise TypeError("document_ids must be a document id or iterable of ids")
    unknown = sorted(set(selected) - set(role_of))
    if unknown:
        raise ReserveAccessError(
            f"{operation}: {len(unknown)} document id(s) are absent from the frozen SGV1 "
            f"role manifest ({', '.join(unknown[:5])}); access fails closed"
        )
    offenders = sorted(
        document_id for document_id in set(selected) if role_of[document_id] == CONFIRMATORY
    )
    if offenders:
        raise ReserveAccessError(
            f"{operation}: {len(offenders)} CONFIRMATORY reserve document(s) selected "
            f"({', '.join(offenders[:5])}). Images, annotations, OCR, candidates, crops, "
            "and verifier scores remain locked until the final SGV1 freeze and unlock."
        )
