"""Fail-closed dataset qualification decisions shared by scripts and tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

__all__ = ["C1Status", "derive_c1_status"]

C1Status = Literal["C1_PASS", "C1_FAIL", "C1_INCONCLUSIVE"]


def derive_c1_status(
    acquisition: Mapping[str, Any] | None,
    prior: Mapping[str, Any] | None,
    pool: Mapping[str, Any] | None,
    duplicates: Mapping[str, Any] | None,
    scan: Mapping[str, Any] | None,
    overlap: Mapping[str, Any] | None,
) -> tuple[C1Status, list[str]]:
    """Derive C1 without accepting a caller-supplied verdict.

    Decisive evidence of corruption/overlap is FAIL. Missing acquisition, licence, or
    audit evidence is INCONCLUSIVE. PASS exists only at the final all-gates branch.
    """
    if acquisition is None:
        return "C1_INCONCLUSIVE", ["missing_acquisition_audit"]

    failures = acquisition.get("failures", [])
    failure_kinds = {
        str(row.get("kind"))
        for row in failures
        if isinstance(row, Mapping) and row.get("kind") is not None
    }
    decisive = {
        "checksum_mismatch",
        "size_mismatch",
        "source_snapshot_mismatch",
        "source_verification_failed",
        "raw_data_tracked",
    }
    if failure_kinds & decisive:
        return "C1_FAIL", sorted(failure_kinds & decisive)

    licence = acquisition.get("license")
    if not isinstance(licence, Mapping) or not licence.get("registry_matches_source", False):
        return "C1_INCONCLUSIVE", ["license_not_established"]
    if not acquisition.get("verified", False):
        return "C1_INCONCLUSIVE", sorted(failure_kinds) or ["acquisition_incomplete"]

    named_inputs = {
        "prior": prior,
        "pool": pool,
        "duplicates": duplicates,
        "scan": scan,
        "overlap": overlap,
    }
    missing = [name for name, value in named_inputs.items() if value is None]
    if missing:
        return "C1_INCONCLUSIVE", [f"missing_{name}" for name in missing]
    assert prior is not None and pool is not None and duplicates is not None
    assert scan is not None and overlap is not None

    if overlap.get("id_overlap") or overlap.get("exact_same_document_pool_ids"):
        return "C1_FAIL", ["historical_document_overlap"]
    if not scan.get("clean", False):
        return "C1_FAIL", ["prior_artifact_hit"]
    if not (
        prior.get("n_documents") == prior.get("expected_prior_universe") == 401
        and not prior.get("duplicate_ids")
        and not prior.get("missing_from_expected")
    ):
        return "C1_INCONCLUSIVE", ["prior_universe_incomplete"]
    if not (
        pool.get("n_documents") == 900
        and pool.get("documents_by_split") == pool.get("expected_documents_by_split")
        and not pool.get("duplicate_ids")
        and pool.get("identity_signals_complete")
        and pool.get("extracted_files") == 1800
        and pool.get("pre_role_access_ids_all_present")
    ):
        return "C1_FAIL", ["pool_census_or_identity_failure"]
    if not duplicates.get("complete", False):
        return "C1_INCONCLUSIVE", ["duplicate_audit_incomplete"]
    return "C1_PASS", []
