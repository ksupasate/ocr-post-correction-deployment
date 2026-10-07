"""Red-team figure provenance gates; never fit or evaluate a scientific model."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/plot_paper3_vf2_figures.py"
SPEC = importlib.util.spec_from_file_location("vf2_figure_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_archive_without_git_metadata_can_record_file_hash_provenance(monkeypatch):
    monkeypatch.setattr(
        replay.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=128, stdout="", stderr="no repository"),
    )
    assert replay.git_head() is None


def test_archive_without_git_executable_can_record_file_hash_provenance(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(replay.subprocess, "run", missing_git)
    assert replay.git_head() is None


@pytest.fixture(scope="module")
def evidence():
    if not (replay.ROOT / replay.REGISTRY).is_file():
        pytest.skip("Frozen corrected VF2 artifact bundle is required for provenance checks")
    return replay.Evidence()


def test_changed_csv_cannot_supply_a_figure_value(evidence):
    path = str(replay.RUN / "tables/ranking_rows.csv")
    records = evidence.csv_records[path]
    key = ("VF2|current_to_qwen|a0_zero_shot|0|0", replay.PRIMARY, "original", "all")
    original = records[key]["harm_auroc"]
    try:
        records[key]["harm_auroc"] = "0.1"
        with pytest.raises(ValueError, match="CSV/registry conflict"):
            evidence.draw_values(
                "current_to_qwen", "0", replay.PRIMARY, "original", "all", "harm_auroc"
            )
    finally:
        records[key]["harm_auroc"] = original


def test_missing_draw_cannot_be_replaced_or_silently_dropped(evidence):
    original = evidence.draws
    try:
        evidence.draws = [
            x
            for x in original
            if not (
                x["value"].get("arm") == "m1_full_adaptation"
                and x["direction"] == "current_to_qwen"
                and str(x["budget"]) == "10"
                and x["draw"] == 3
                and x["population"] == "original"
                and x["harm_definition"] == replay.PRIMARY
                and x["value"].get("unit") == "all"
            )
        ]
        with pytest.raises(ValueError, match="incomplete predetermined draws"):
            evidence.draw_values(
                "current_to_qwen", "10", replay.PRIMARY, "original", "all", "harm_auroc"
            )
    finally:
        evidence.draws = original


def test_missing_registered_interval_cannot_be_fabricated(evidence):
    prefix = evidence.prefix("current_to_qwen", "pool", replay.PRIMARY, "original", "winner")
    key = prefix + "|ci_low"
    original = evidence.entries.pop(key)
    try:
        with pytest.raises(ValueError, match="aggregate record not found"):
            evidence.ranking(
                4, "a", "current_to_qwen", "pool", replay.PRIMARY, "original", "winner"
            )
    finally:
        evidence.entries[key] = original


def test_duplicate_plot_observations_fail_loudly():
    rows = [{"figure": "Fig3", "policy": "plug_in", "value": 0.2}] * 2
    with pytest.raises(ValueError, match="plot record not unique"):
        replay.select(rows, figure="Fig3", policy="plug_in")


def test_no_action_cannot_be_promoted_to_a_risk_coordinate(evidence):
    original = evidence.audit
    try:
        evidence.audit = []
        evidence.load_figure_data()
        row = replay.select(
            evidence.audit,
            figure="Fig3",
            direction="current_to_qwen",
            policy="conservative",
            metric="median_harm",
        )
        row["plot_status"] = "plotted"
        row["value"] = 0.0
        with pytest.raises(ValueError, match="no action plotted as zero harm"):
            evidence.validate()
    finally:
        evidence.audit = original
