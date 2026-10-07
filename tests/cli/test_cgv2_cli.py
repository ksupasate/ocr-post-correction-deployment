"""The CGV2 CLI commands exercised against a real synthetic corpus.

``cgv2 study`` is the producer of every CGV2 artifact; ``cgv2 tables`` and ``cgv2 gates``
are its consumers. Running the three in order over a rendered corpus, and asserting what
each wrote, is the executable evidence that the phase's artifact chain hangs together --
the same discipline as ``test_command_surface``, scoped to CGV2 so the module-scoped
chain there stays untouched.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from ocr_risk.cli.app import app
from ocr_risk.cli.cmd_cgv2 import _contrast_rows
from ocr_risk.io.hashing import file_sha256
from ocr_risk.io.paths import project_root

CONFIG = "configs/experiments/smoke_synthetic.yaml"
CORPUS = "datasets=[{id: synthetic, enabled: true, params: {n_documents: 10, seed: 4242}}]"


@pytest.mark.parametrize(
    ("bootstrap_unit", "multiplicity", "message"),
    [
        ("site", "holm", "bootstrap_unit=document"),
        ("document", "bonferroni", "multiplicity=holm"),
    ],
)
def test_contrasts_reject_unimplemented_statistical_methods(
    bootstrap_unit: str, multiplicity: str, message: str
) -> None:
    with pytest.raises(typer.BadParameter, match=message):
        _contrast_rows(
            [],
            [],
            "strict",
            bootstrap_unit=bootstrap_unit,
            multiplicity=multiplicity,
        )


@pytest.fixture(scope="module")
def cgv2(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, object]]:
    """Render a tiny corpus, run the stages the study reads, then the CGV2 commands."""
    root = tmp_path_factory.mktemp("cgv2")
    saved = {
        var: os.environ.get(var)
        for var in (
            "OCR_RISK_DATA_ROOT",
            "OCR_RISK_ARTIFACT_ROOT",
            "OCR_RISK_CACHE_DIR",
            "OCR_RISK_MANIFEST_ROOT",
        )
    }
    for var, sub in (
        ("OCR_RISK_DATA_ROOT", "data"),
        ("OCR_RISK_MANIFEST_ROOT", "manifests"),
        ("OCR_RISK_ARTIFACT_ROOT", "artifacts"),
        ("OCR_RISK_CACHE_DIR", "cache"),
    ):
        (root / sub).mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(root / sub)

    runner = CliRunner()
    config = str(project_root() / CONFIG)

    def invoke(*args: str) -> object:
        result = runner.invoke(app, list(args), catch_exceptions=False)
        assert result.exit_code == 0, (
            f"`{' '.join(args)}` exited {result.exit_code}\n{result.output}"
        )
        return result

    try:
        for args in (
            ("data", "synth", "--pages", "10", "--seed", "4242"),
            ("data", "manifest", config, "--set", CORPUS),
            ("ocr", "run", config, "--set", CORPUS),
            ("ocr", "canonicalize", config, "--set", CORPUS),
            ("align", "run", config, "--set", CORPUS),
            ("sites", "build", config, "--set", CORPUS),
            ("candidates", "generate", config, "--set", CORPUS),
            ("experiment", "run", "-c", config, "--set", CORPUS),
        ):
            invoke(*args)

        out = root / "cgv2"
        study_args = (
            "cgv2",
            "study",
            config,
            "--rungs",
            "g0_lexical,g3_edit_aware,g5_structural,g6_union",
            "--role",
            "calibrate",
            "--out",
            str(out),
        )
        invoke(*study_args)
        study_mtime = (out / "cgv2_study.json").stat().st_mtime_ns
        resumed = invoke(*study_args)
        assert "verified complete CGV2 run" in str(getattr(resumed, "output", ""))
        assert (out / "cgv2_study.json").stat().st_mtime_ns == study_mtime
        invoke(
            *study_args,
            "--pilot-engine",
            "synth_a",
            "--pilot-documents-per-dataset",
            "1",
        )
        invoke(
            "analyze",
            "alignment-twin",
            "--experiment",
            "smoke_synthetic",
            "--role",
            "calibrate",
            "--out-dir",
            str(out),
        )
        invoke("cgv2", "tables", "--source", str(out), "--role", "calibrate")
        invoke(
            "cgv2",
            "tables",
            "--source",
            str(out),
            "--role",
            "calibrate",
            "--sensitivity",
            "no_twin",
        )
        invoke(
            "cgv2",
            "tables",
            "--source",
            str(out),
            "--role",
            "calibrate",
            "--harm-policy",
            "non_improving",
        )
        # A clean synthetic review file: no findings, construction certified. The real
        # review (docs/cgv2/integrity_review.json) must never bleed into a smoke verdict.
        findings = out / "integrity_findings.json"
        findings.write_text(
            json.dumps(
                {
                    "schema_version": "cgv2-integrity-review-v1",
                    "reviewed_at": "2026-08-23",
                    "certifications": {"gt_blind_natural_site_construction": True},
                    "findings": [],
                }
            )
            + "\n"
        )
        # Affirmative freshness evidence: a prior study record that scored only the
        # calibrate role, so this experiment's evaluate documents were never touched.
        prior = root / "prior-study.json"
        prior.write_text(
            json.dumps({"role_scored": "calibrate", "experiment": "smoke_synthetic"}) + "\n"
        )
        invoke(
            "cgv2",
            "integrity",
            "--source",
            str(out),
            "--role",
            "calibrate",
            "--findings",
            str(findings),
            "--prior-evaluate-study",
            str(prior),
        )
        invoke("cgv2", "gates", "--source", str(out), "--role", "calibrate")
        yield {"root": root, "out": out, "runner": runner}
    finally:
        for var, previous in saved.items():
            if previous is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = previous


def _read(out: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(out / name, keep_default_na=False)


def test_study_writes_every_canonical_artifact(cgv2: dict[str, object]) -> None:
    out = cgv2["out"]  # type: ignore[assignment]
    record = json.loads((out / "cgv2_study.json").read_text())  # type: ignore[operator]
    assert record["role_scored"] == "calibrate"
    assert record["protocol_frozen_commit"]
    assert record["protocol_amendment_commit"]
    assert len(record["protocol_sha256"]) == 64
    assert record["k_grid"] == [1, 2, 4, 8]
    assert record["headline_datasets"] == ["synthetic"]
    assert record["stress_track_datasets"] == []
    assert record["headline_total_candidates"] == record["total_candidates"]
    assert record["bootstrap"] == {
        "ci_level": 0.95,
        "multiplicity": "holm",
        "n_resamples": 400,
        "seed": 7,
        "unit": "document",
    }
    certificate_path = out / "leakage_certificates.json"  # type: ignore[operator]
    certificates = json.loads(certificate_path.read_text())
    assert certificates["overall_pass"] is True
    assert certificates["partition_sha256"] == record["partition_sha256"]
    assert record["leakage_certificates_file"] == certificate_path.name
    assert len(record["leakage_certificates_sha256"]) == 64
    assert record["duration_seconds"] >= 0
    assert record["peak_rss_bytes"] is None or record["peak_rss_bytes"] > 0
    assert record["runtime_seconds"]["total"] >= 0
    assert record["runtime_by_generator_seconds"]
    assert record["total_candidates"] == record["site_proposals"] + record["region_proposals"]
    completion = json.loads((out / "cgv2_completion.json").read_text())  # type: ignore[operator]
    assert completion["status"] == "complete"
    assert completion["exit_code"] == 0
    assert completion["expected_artifact_count"] == len(completion["outputs"])
    # Every rung's fold certificate must be present and clean for every engine.
    for engine_id, certificate in record["certificates"].items():
        assert certificate["pass"] is True, engine_id
        assert certificate["held_out_engine"] == engine_id
        assert certificate["sources_clean"] is True
        for rung, entry in certificate["rungs"].items():
            assert entry.get("clean", False), f"{engine_id}/{rung} leaked"
    for name in (
        "cgv2_site_census.csv",
        "cgv2_proposals.csv",
        "cgv2_region_proposals.csv",
        "cgv2_source_geometry.csv",
        "cgv2_fold_lexicons.csv",
        "cgv2_contrasts.csv",
        "cgv2_oracle.csv",
    ):
        assert (out / name).exists(), name  # type: ignore[operator]

    proposals = _read(out, "cgv2_proposals.csv")  # type: ignore[arg-type]
    assert {
        "candidate_id",
        "document_id",
        "dataset_id",
        "engine_id",
        "source_char_start",
        "source_char_end",
        "source_span_ids",
        "alignment_component_ids",
        "candidate_text",
        "operation_type",
        "generator_id",
        "generator_version",
        "generator_config_sha256",
        "generator_score",
        "generator_rank",
        "fold_id",
        "candidate_budget_k",
        "ambiguity_status",
        "pool",
        "generator_provenance",
        "source_geometry_id",
    } <= set(proposals.columns)
    assert set(proposals["pool"]) == {"natural"}
    geometry = _read(out, "cgv2_source_geometry.csv")  # type: ignore[arg-type]
    assert {
        "source_geometry_id",
        "source_bbox",
        "source_bbox_normalized",
        "source_span_geometry",
        "crop_recipe",
        "geometry_source_run",
    } <= set(geometry.columns)
    assert record["source_geometry_rows"] == len(geometry)


def test_pilot_has_a_separate_bounded_and_resumable_identity(cgv2: dict[str, object]) -> None:
    out = cgv2["out"]  # type: ignore[assignment]
    study = json.loads((out / "cgv2_study__pilot_synth_a.json").read_text())  # type: ignore[operator]
    assert study["pilot"] is True
    assert study["pilot_engine"] == "synth_a"
    assert study["pilot_documents_per_dataset"] == 1
    assert study["engines"] == ["synth_a"]
    assert study["scored_documents"] == 1
    assert study["datasets"] == ["synthetic"]
    assert study["completion_marker"] == "cgv2_completion__pilot_synth_a.json"
    completion = json.loads(
        (out / "cgv2_completion__pilot_synth_a.json").read_text()  # type: ignore[operator]
    )
    assert completion["status"] == "complete"


def test_contrasts_carry_holm_adjustment_and_per_engine_rows(cgv2: dict[str, object]) -> None:
    contrasts = _read(cgv2["out"], "cgv2_contrasts.csv")  # type: ignore[arg-type]
    primary = contrasts[contrasts["metric"] == "site_availability"]
    assert len(primary) == 4  # one row per synthetic engine
    assert {"p_holm", "reject_holm"} <= set(primary.columns)
    assert set(contrasts["n_resamples"]) == {400}
    assert set(contrasts["bootstrap_seed"]) == {7}
    assert set(contrasts["ci_level"]) == {0.95}
    assert set(contrasts["bootstrap_unit"]) == {"document"}
    sensitivity = _read(cgv2["out"], "cgv2_contrasts__no_twin.csv")  # type: ignore[arg-type]
    assert set(sensitivity["metric"]) == set(contrasts["metric"])
    for metric in (
        "site_availability",
        "structural_stratum_availability",
        "exact_only_availability",
    ):
        assert {"p_holm", "reject_holm"} <= set(
            sensitivity[sensitivity["metric"] == metric].columns
        )
    assert (cgv2["out"] / "cgv2_oracle__no_twin.csv").exists()  # type: ignore[operator]
    assert (cgv2["out"] / "cgv2_analysis_manifest__no_twin.json").exists()  # type: ignore[operator]


def test_harm_policy_sensitivity_is_materialized(cgv2: dict[str, object]) -> None:
    quality = _read(cgv2["out"], "cgv2_quality__non_improving.csv")  # type: ignore[arg-type]
    assert set(quality["harm_policy"]) == {"non_improving"}
    manifest = json.loads(
        (cgv2["out"] / "cgv2_analysis_manifest__non_improving.json").read_text()  # type: ignore[operator]
    )
    assert manifest["harm_policy"] == "non_improving"


def test_twin_audit_writes_both_views(cgv2: dict[str, object]) -> None:
    out = cgv2["out"]  # type: ignore[assignment]
    audits = Path(out) / "audits" / "calibrate"  # type: ignore[arg-type]
    twins = pd.read_csv(audits / "alignment_twin.csv", keep_default_na=False)
    assert {"engine_id", "site_kind", "n_evaluable_sites", "n_twin_signature"} <= set(twins.columns)
    census = pd.read_csv(audits / "region_census.csv", keep_default_na=False)
    assert {
        "eligible",
        "degenerate",
        "ambiguous_member",
        "n_pairs_before_exclusion",
        "n_pairs_eligible_after_exclusion",
    } <= set(census.columns)
    assert (audits / "region_pair_audit.csv").exists()
    components = pd.read_csv(audits / "alignment_component_audit.csv", keep_default_na=False)
    assert {
        "operation",
        "state",
        "punctuation_boundary",
        "whitespace_boundary",
        "ocr_empty_anchor",
        "gt_empty_anchor",
    } <= set(components.columns)
    site_eligibility = pd.read_csv(audits / "alignment_site_audit.csv", keep_default_na=False)
    assert set(site_eligibility["state"]) <= {"eligible", "ambiguous", "excluded", "unresolved"}
    site_projection = pd.read_csv(audits / "site_projection_audit.csv", keep_default_na=False)
    assert site_projection["projection_correct"].astype(bool).all()
    region_projection = pd.read_csv(audits / "region_projection_audit.csv", keep_default_na=False)
    assert region_projection.empty or region_projection["projection_correct"].astype(bool).all()
    gate = json.loads((audits / "alignment_audit_gate.json").read_text())
    assert gate["overall_pass"] is True
    assert gate["role"] == "calibrate"
    # Every audit output is hash-bound so downstream stages can refuse a stale audit.
    assert gate["outputs"]["alignment_site_audit.csv"]["sha256"] == file_sha256(
        audits / "alignment_site_audit.csv"
    )


def test_tables_derive_from_the_canonical_csvs(cgv2: dict[str, object]) -> None:
    out = cgv2["out"]  # type: ignore[assignment]
    # Ten synthetic pages may legitimately hold no repairable proposal, so opportunity
    # and preservation may be empty -- but never column-less, because downstream code
    # filters on the columns. Quality, budget, and taxonomy always have rows.
    for name in (
        "cgv2_opportunity.csv",
        "cgv2_opportunity_by_dataset.csv",
        "cgv2_structural_coverage.csv",
        "cgv2_structural_operations.csv",
        "cgv2_region_opportunity.csv",
        "cgv2_preservation.csv",
    ):
        table = _read(out, name)  # type: ignore[index]
        assert not table.columns.empty, name
    for name in (
        "cgv2_quality.csv",
        "cgv2_quality_by_dataset.csv",
        "cgv2_budget.csv",
        "cgv2_failure_taxonomy.csv",
    ):
        table = _read(out, name)  # type: ignore[index]
        assert not table.empty, name


def test_gates_read_only_the_artifacts(cgv2: dict[str, object]) -> None:
    """On calibrate-role artifacts without an integrity review the gate stays open about it."""
    from ocr_risk.experiments.cgv2_gates import evaluate_cgv2_gates

    result = evaluate_cgv2_gates(cgv2["out"], role="calibrate")  # type: ignore[arg-type]
    assert result["verdict"] in {
        "GENERATOR READY",
        "GENERATOR PARTIALLY READY",
        "GENERATOR NOT READY",
        "INCONCLUSIVE",
    }
    # CG1-CG3 plus the integrity review plus CG4; H2 gains the validity criterion G.
    assert len(result["criteria"]) == 5
    assert len(result["h2_readiness"]["criteria"]) == 7
    generator = json.loads(
        (cgv2["out"] / "generator_gate__calibrate.json").read_text()  # type: ignore[operator]
    )
    h2 = json.loads(
        (cgv2["out"] / "h2_readiness_gate__calibrate.json").read_text()  # type: ignore[operator]
    )
    assert generator["verdict"] == result["verdict"]
    assert h2["verdict"] == result["h2_readiness"]["verdict"]
    criteria = {c["key"]: c["met"] for c in result["criteria"]}
    assert criteria["VALIDITY_integrity_review"] is True


def _unwrapped(output: str) -> str:
    """CLI output with Rich's box drawing and line wrapping collapsed to single spaces.

    The error text is rendered into an 80-column panel, so where it breaks depends on how
    long the temporary directory in the message happens to be -- which makes a substring
    assertion against the raw output pass or fail on the pytest tmp path. Collapsing the
    frame and the whitespace tests the same claim without that dependence.
    """
    stripped = "".join(character for character in output if character not in "│╭╮╰╯─")
    return " ".join(stripped.split())


def test_gates_refuse_to_run_without_the_integrity_gate(cgv2: dict[str, object]) -> None:
    """Stage ordering is enforced fail-closed: no validity review, no verdict."""
    runner: CliRunner = cgv2["runner"]  # type: ignore[assignment]
    out = Path(str(cgv2["out"]))
    integrity = out / "cgv2_integrity_gate.json"
    withheld = out / "withheld_integrity_gate.json"
    integrity.rename(withheld)
    try:
        result = runner.invoke(app, ["cgv2", "gates", "--source", str(out), "--role", "calibrate"])
        assert result.exit_code != 0
        assert "cgv2 integrity" in _unwrapped(result.output)
    finally:
        withheld.rename(integrity)


def test_tables_refuse_to_run_without_the_study(cgv2: dict[str, object]) -> None:
    runner: CliRunner = cgv2["runner"]  # type: ignore[assignment]
    result = runner.invoke(
        app,
        ["cgv2", "tables", "--source", str(cgv2["root"] / "empty"), "--role", "evaluate"],
    )
    assert result.exit_code != 0
    assert "missing study artifacts" in _unwrapped(result.output)
