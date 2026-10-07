"""SYNTHETIC provenance fixtures; no scientific outcomes computed."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "vf2_reporting_v2", ROOT / "scripts/paper3_vf2_reporting_v2.py"
)
assert SPEC and SPEC.loader
reporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)


def setup_runs(tmp_path, monkeypatch, change=False, bad_hash=False):
    ex = reporting.ex
    monkeypatch.setattr(ex, "OUT", tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"determinism_atol": 1e-12, "determinism_rtol": 1e-12}))
    monkeypatch.setattr(ex, "CONFIG", config)
    for run in [1, 2]:
        rd = tmp_path / f"run{run}"
        (rd / "tables").mkdir(parents=True)
        (rd / "figures").mkdir()
        source = rd / "tables/scientific_results.json"
        source.write_text(
            json.dumps(
                {"synthetic": True, "run": run, "fixture_value": 2 if change and run == 2 else 1}
            )
        )
        manifest = {
            "source_sha256": {"tables/scientific_results.json": ex.sha(source)},
            "analysis_only": True,
        }
        if bad_hash and run == 2:
            manifest["source_sha256"]["tables/scientific_results.json"] = "0" * 64
        (rd / "figures/VF2-1_manifest.json").write_text(json.dumps(manifest))


def test_administrative_source_hash_difference_accepts_exact_data(tmp_path, monkeypatch):
    setup_runs(tmp_path, monkeypatch)
    report = reporting.verify_determinism()
    assert report["status"] == "BIT-IDENTICAL"
    assert any(
        c["result"] == "IDENTICAL SOURCE DATA; RUN-SPECIFIC PROVENANCE HASHES"
        for c in report["checks"]
    )


def test_numeric_source_difference_is_rejected(tmp_path, monkeypatch):
    setup_runs(tmp_path, monkeypatch, change=True)
    with pytest.raises(AssertionError):
        reporting.verify_determinism()
    assert not (tmp_path / "determinism.json").exists()


def test_false_source_hash_is_rejected(tmp_path, monkeypatch):
    setup_runs(tmp_path, monkeypatch, bad_hash=True)
    with pytest.raises(AssertionError):
        reporting.verify_determinism()
    assert not (tmp_path / "determinism.json").exists()
