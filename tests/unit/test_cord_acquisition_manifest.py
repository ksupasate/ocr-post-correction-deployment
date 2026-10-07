from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocr_risk.datasets.cord import shard_document_plan

REPO = Path(__file__).resolve().parents[2]


def test_cord_archive_pins_match_the_upstream_snapshot() -> None:
    manifest = json.loads((REPO / "manifests/datasets/cord.json").read_text(encoding="utf-8"))
    snapshot = json.loads(
        (REPO / "manifests/sgv1/cord_upstream_snapshot.json").read_text(encoding="utf-8")
    )
    upstream = {entry["path"]: entry for entry in snapshot["tree"]["files"]}
    assert len(manifest["archives"]) == len(upstream) == 6
    for archive in manifest["archives"]:
        upstream_path = "data/" + archive["url"].split("/data/", 1)[1]
        assert archive["checksum_kind"] == "upstream_verified_checksum"
        assert archive["sha256"] == upstream[upstream_path]["lfs_oid"]
        assert archive["size_bytes"] == upstream[upstream_path]["size_bytes"]


def test_cord_expected_rows_match_the_upstream_size_census() -> None:
    manifest = json.loads((REPO / "manifests/datasets/cord.json").read_text(encoding="utf-8"))
    snapshot = json.loads(
        (REPO / "manifests/sgv1/cord_upstream_snapshot.json").read_text(encoding="utf-8")
    )
    for split, record in snapshot["size_census"]["splits"].items():
        assert manifest["expected"][f"{split}_documents"] == record["rows"]


def test_train_shard_offsets_are_deterministic_and_global() -> None:
    assert shard_document_plan([200, 200, 200, 200]) == [0, 200, 400, 600]


def test_negative_shard_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        shard_document_plan([200, -1])


def test_pre_role_preview_ids_are_confirmatory_ineligible() -> None:
    exclusions = json.loads(
        (REPO / "manifests/sgv1/pre_role_access_exclusions.json").read_text(encoding="utf-8")
    )
    event = exclusions["events"][0]
    assert len(event["document_ids"]) == 26
    assert event["disposition"]["confirmatory_eligible"] is False
    assert "CONFIRMATORY" not in event["disposition"]["allowed_roles"]
