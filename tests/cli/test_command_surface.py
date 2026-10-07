"""Every CLI command exercised against a real synthetic corpus.

The CLI is the research entrypoint: a stage is run by typing a command, and a command that
wires the wrong config field, reads the wrong artifact, or reports the wrong count produces
a plausible-looking result rather than an error. The pilot already lost 109,305 decision
rows to exactly that class of defect. Importing the module proves none of it; these tests
run the commands in dependency order over a rendered corpus and assert what each one wrote.

One module-scoped fixture drives the whole chain, because the commands are a pipeline and
re-running the earlier stages per test would cost minutes for no extra coverage.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from ocr_risk.cli.app import app
from ocr_risk.io.paths import project_root

CONFIG = "configs/experiments/smoke_synthetic.yaml"
CORPUS = "datasets=[{id: synthetic, enabled: true, params: {n_documents: 10, seed: 4242}}]"


@pytest.fixture(scope="module")
def cli(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, object]]:
    """Run every stage command once, in order, into an isolated tree."""
    root = tmp_path_factory.mktemp("cli")
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
    rh1_config = str(project_root() / "configs/experiments/smoke_rh1.yaml")
    results = root / "results"

    def invoke(*args: str) -> object:
        if args[:2] == ("sites", "baseline"):
            # This legacy CLI command writes under project_root rather than an env root.
            # Keep its synthetic output inside the test fixture, outside release evidence.
            from ocr_risk.cli import cmd_sites

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(cmd_sites, "project_root", lambda: root)
                result = runner.invoke(app, list(args), catch_exceptions=False)
            assert (root / "results/generated/raw_ocr_baseline.csv").is_file()
        else:
            result = runner.invoke(app, list(args), catch_exceptions=False)
        assert result.exit_code == 0, (
            f"`{' '.join(args)}` exited {result.exit_code}\n{result.output}"
        )
        return result

    outputs: dict[str, str] = {}
    chain = (
        ("synth", ("data", "synth", "--pages", "10", "--seed", "4242")),
        ("manifest", ("data", "manifest", config, "--set", CORPUS)),
        ("ocr", ("ocr", "run", config, "--set", CORPUS)),
        ("canonicalize", ("ocr", "canonicalize", config, "--set", CORPUS)),
        ("align", ("align", "run", config, "--set", CORPUS)),
        ("align_stats", ("align", "stats", config, "--set", CORPUS)),
        ("sites", ("sites", "build", config, "--set", CORPUS)),
        ("baseline", ("sites", "baseline", config, "--set", CORPUS)),
        ("candidates", ("candidates", "generate", config, "--set", CORPUS)),
        ("experiment", ("experiment", "run", "-c", config, "--set", CORPUS)),
        ("status", ("experiment", "status", "--experiment", "smoke_synthetic")),
        (
            "report",
            ("analyze", "report", "--experiment", "smoke_synthetic", "--out", str(results)),
        ),
        (
            "gate",
            ("gate", "pilot", "--experiment", "smoke_synthetic", "--results", str(results)),
        ),
        ("provenance", ("provenance", str(results / "figure_manifest.json"))),
        ("provenance_json", ("provenance", str(results / "figure_manifest.json"), "--json")),
        ("runs", ("provenance", "list")),
        ("leakage", ("audit", "leakage", "--experiment", "smoke_synthetic")),
        ("licenses", ("audit", "licenses")),
        ("docs", ("audit", "docs")),
        ("engines", ("ocr", "engines")),
        ("datasets", ("data", "list")),
        ("version", ("version",)),
        (
            "generator_study",
            (
                "candidates",
                "study",
                config,
                "--set",
                CORPUS,
                "--generators",
                "g0_lexical,g1_error_gated,g3_edit_aware",
                "--out",
                str(root / "generators"),
            ),
        ),
        (
            "generator_study_evaluate",
            (
                "candidates",
                "study",
                config,
                "--set",
                CORPUS,
                "--generators",
                "g0_lexical,g3_edit_aware",
                "--role",
                "evaluate",
                "--out",
                str(root / "generators"),
            ),
        ),
        ("rh1", ("experiment", "rh1", "-c", rh1_config, "--source", "smoke_synthetic")),
        (
            "analyze_rh1",
            (
                "analyze",
                "rh1",
                "--experiment",
                "smoke_rh1",
                "--pool",
                "smoke_synthetic",
                "--results",
                str(results),
            ),
        ),
        (
            "rh1_gate",
            (
                "gate",
                "rh1",
                "--table",
                str(results / "rh1_discrimination.csv"),
                "--verifier",
                "v6_full",
                "--out",
                str(results),
            ),
        ),
        (
            "generator_gate",
            ("gate", "generator", "--study", str(root / "generators"), "--out", str(results)),
        ),
        (
            "recovery_figures",
            (
                "analyze",
                "recovery",
                "--study",
                str(root / "generators"),
                "--rh1-table",
                str(results / "rh1_discrimination.csv"),
                "--rh1",
                "smoke_rh1",
                "--pool",
                "smoke_synthetic",
                "--out",
                str(results),
            ),
        ),
        (
            "recovery_tables",
            (
                "analyze",
                "recovery-tables",
                "--study",
                str(root / "generators"),
                "--rh1-table",
                str(results / "rh1_discrimination.csv"),
                "--gates",
                str(results),
                "--out",
                str(results),
            ),
        ),
        (
            "h2_gate",
            (
                "gate",
                "h2-readiness",
                "--generator",
                "g3_edit_aware",
                "--study",
                str(root / "generators"),
                "--rh1",
                "smoke_rh1",
                "--out",
                str(results),
            ),
        ),
    )
    for name, args in chain:
        outputs[name] = invoke(*args).output  # type: ignore[attr-defined]

    try:
        yield {"root": root, "results": results, "runner": runner, "config": config, "out": outputs}
    finally:
        for var, previous in saved.items():
            if previous is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = previous


def _out(cli: dict[str, object], key: str) -> str:
    return str(cli["out"][key])  # type: ignore[index]


def test_the_whole_stage_chain_runs_from_the_command_line(cli: dict[str, object]) -> None:
    """Not a tautology: the fixture asserts exit code 0 on each of 22 invocations."""
    assert len(cli["out"]) == 31  # type: ignore[arg-type]


def test_synthetic_output_is_announced_at_the_terminal_not_only_in_the_artifact(
    cli: dict[str, object],
) -> None:
    """A researcher reads the terminal. The label has to be where they are looking."""
    assert "SYNTHETIC" in _out(cli, "synth")
    assert "SYNTHETIC" in _out(cli, "report")


def test_alignment_stats_report_ambiguity_per_engine(cli: dict[str, object]) -> None:
    """Per-engine ambiguity is a reported confound, so the command must break it out."""
    text = _out(cli, "align_stats")
    assert "ambiguity" in text.lower()
    assert text.count("synth_") >= 2, "one row per engine, not a pooled total"


def test_the_report_command_lists_the_tables_and_figures_it_wrote(
    cli: dict[str, object],
) -> None:
    results = Path(str(cli["results"]))
    manifest = json.loads((results / "figure_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"], "a manifest with no outputs cannot make anything traceable"
    for name, entry in manifest["outputs"].items():
        assert (results / name).is_file(), f"{name} is in the manifest but not on disk"
        assert len(entry["sha256"]) == 64


def test_the_gate_prints_a_verdict_for_every_hypothesis(cli: dict[str, object]) -> None:
    report = json.loads(
        (Path(str(cli["results"])) / "gate_report.json").read_text(encoding="utf-8")
    )
    assert {v["hypothesis"] for v in report["verdicts"]} == {"H1", "H2", "H3", "H4"}
    assert report["synthetic"] is True
    assert report["harm_policy"] == "strict_worsening"


def test_provenance_walks_back_to_the_manifest_stage(cli: dict[str, object]) -> None:
    assert "manifest" in _out(cli, "provenance")


def test_every_audit_passes_on_the_synthetic_run(cli: dict[str, object]) -> None:
    for key in ("leakage", "licenses", "docs"):
        assert "FAIL" not in _out(cli, key).upper(), f"audit {key} reported a failure"


def test_the_determinism_audit_flags_a_stage_whose_two_runs_disagree(
    cli: dict[str, object],
) -> None:
    """The tree holds two writers per stage: the ad-hoc commands and the pipeline.

    They are not the same procedure -- ``candidates generate`` fits its lexicon over every
    site while the split-aware runner fits it on the fit split only -- so their digests
    must differ, and the audit must say so and exit non-zero. An audit that passed here
    would be an audit that cannot detect anything.

    They share a config hash and a source hash, so they fall in the same comparison group.
    That is the point of the grouping: runs of *different* code are allowed to differ.
    """
    runner: CliRunner = cli["runner"]  # type: ignore[assignment]
    result = runner.invoke(app, ["audit", "determinism", "--experiment", "smoke_synthetic"])
    assert result.exit_code != 0
    # Rich wraps the table, so match on the summary line rather than a cell.
    assert "produced different outputs across runs of the SAME" in result.output
    assert "NOT COMPARED" in result.output.replace("\n", " "), (
        "a single-run group must not pass vacuously"
    )


# --- failure paths: a command that cannot do its job must say so, not do half of it ----


def test_a_stage_run_before_its_input_exists_is_refused(tmp_path: Path) -> None:
    """``align run`` without a canonicalize run must fail loudly, not write an empty table."""
    saved = {v: os.environ.get(v) for v in ("OCR_RISK_ARTIFACT_ROOT", "OCR_RISK_DATA_ROOT")}
    os.environ["OCR_RISK_ARTIFACT_ROOT"] = str(tmp_path / "artifacts")
    os.environ["OCR_RISK_DATA_ROOT"] = str(tmp_path / "data")
    try:
        runner = CliRunner()
        for args in (
            ("align", "run", str(project_root() / CONFIG)),
            ("sites", "build", str(project_root() / CONFIG)),
            ("candidates", "generate", str(project_root() / CONFIG)),
        ):
            result = runner.invoke(app, list(args))
            assert result.exit_code != 0, f"`{' '.join(args)}` silently succeeded with no input"
    finally:
        for var, previous in saved.items():
            if previous is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = previous


def test_an_unknown_experiment_name_names_the_ones_that_exist() -> None:
    result = CliRunner().invoke(app, ["gate", "pilot", "--experiment", "no_such_experiment"])
    assert result.exit_code != 0
    assert "known" in result.output


def test_the_config_commands_round_trip_a_real_experiment(tmp_path: Path) -> None:
    runner = CliRunner()
    config = str(project_root() / CONFIG)

    schema_path = tmp_path / "schema.json"
    assert runner.invoke(app, ["config", "schema", "--out", str(schema_path)]).exit_code == 0
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "properties" in schema

    shown = runner.invoke(app, ["config", "show", config])
    assert shown.exit_code == 0
    assert "smoke_synthetic" in shown.output

    validated = runner.invoke(app, ["config", "validate", config])
    assert validated.exit_code == 0

    broken = tmp_path / "broken.yaml"
    broken.write_text("name: broken\nrisk: {epsilon_grid: not-a-list}\n", encoding="utf-8")
    assert runner.invoke(app, ["config", "validate", str(broken)]).exit_code != 0


# --- the recovery-phase commands -------------------------------------------------------


def test_the_generator_study_reports_the_oracle_ceiling_as_an_oracle(
    cli: dict[str, object],
) -> None:
    """A deployable-looking name on an unreachable upper bound is how a ceiling gets
    quoted as a result."""
    text = _out(cli, "generator_study")
    assert "GROUND-TRUTH ORACLE" in text
    assert "NOT deployable" in text


def test_scoring_the_held_out_documents_announces_that_it_is_doing_so(
    cli: dict[str, object],
) -> None:
    """The development/confirmation boundary is the discipline; it has to be visible."""
    assert "HELD-OUT" in _out(cli, "generator_study_evaluate")
    assert "HELD-OUT" not in _out(cli, "generator_study")


def test_the_generator_study_writes_a_cell_per_generator_engine_and_corpus(
    cli: dict[str, object],
) -> None:
    directory = Path(str(cli["root"])) / "generators"
    quality = pd.read_csv(directory / "generator_quality.csv")
    assert {"g0_lexical", "g1_error_gated", "g3_edit_aware"} == set(quality["generator_id"])
    assert "ALL" in set(quality["dataset_id"])
    assert (quality["oracle_safe_coverage"] <= 1.0).all()
    proposals = pd.read_csv(directory / "generator_proposals.csv")
    assert not proposals.empty
    assert (proposals["candidate_text"] != proposals["original_ocr"]).all()


def test_the_matched_run_certifies_every_pair_before_reporting_anything(
    cli: dict[str, object],
) -> None:
    text = _out(cli, "rh1")
    assert '"all_matched": true' in text
    assert "match certificates" in text


def test_the_matched_certificates_are_written_beside_the_predictions(
    cli: dict[str, object],
) -> None:
    from ocr_risk.io.artifacts import ArtifactStore
    from ocr_risk.schemas.enums import StageName

    store = ArtifactStore()
    predict_run = store.latest(StageName.PREDICT, "smoke_rh1")
    assert predict_run is not None
    payload = store.read_json(predict_run, "match_certificates.json")
    assert payload["all_matched"] is True
    assert payload["n_pairs"] == 12  # 4 target engines x 3 donors
    for certificate in payload["certificates"]:
        left, right = certificate["zero_shot"], certificate["reference"]
        assert left["n_fit_candidates"] == right["n_fit_candidates"]
        assert left["n_fit_harmful"] == right["n_fit_harmful"]
        assert certificate["held_out_engine"] not in left["fit_engines"]
        assert certificate["held_out_engine"] in right["fit_engines"]


def test_the_rh1_table_reports_ranking_endpoints_not_calibration_ones(
    cli: dict[str, object],
) -> None:
    from ocr_risk.analysis.rh1_discrimination import RH1_METRICS

    table = pd.read_csv(Path(str(cli["results"])) / "rh1_discrimination.csv")
    assert set(table["metric"]) == set(RH1_METRICS)
    assert "matched_in_engine" in table.columns
    assert (table["n_candidates"] > 0).all()
    # Both arms scored the same candidates, so the safe base rate must agree exactly.
    for _, group in table.groupby(["held_out_engine", "donor_engine", "verifier_id"]):
        assert group["base_rate_safe"].nunique() == 1


def test_the_h2_gate_emits_one_of_its_three_verdicts_and_shows_its_working(
    cli: dict[str, object],
) -> None:
    payload = json.loads(
        (Path(str(cli["results"])) / "h2_readiness_gate.json").read_text(encoding="utf-8")
    )
    assert payload["verdict"] in {"H2 READY", "H2 NOT READY", "H2 INCONCLUSIVE"}
    assert {c["key"] for c in payload["criteria"]} == {
        "A_repair_opportunity",
        "B_oracle_safe_coverage",
        "C_sample_size",
        "D_overcorrection_observable",
        "E_natural_pool",
        "F_no_unresolved_confound",
    }
    for criterion in payload["criteria"]:
        assert criterion["threshold"]
        assert criterion["observed"]


def test_the_rh1_gate_reads_its_verdict_from_the_named_primary_endpoint(
    cli: dict[str, object],
) -> None:
    payload = json.loads((Path(str(cli["results"])) / "rh1_gate.json").read_text(encoding="utf-8"))
    assert payload["verdict"] in {
        "DISCRIMINATION DEGRADATION SUPPORTED",
        "PARTIALLY SUPPORTED",
        "NOT SUPPORTED",
        "INCONCLUSIVE",
    }
    assert payload["criteria"][0]["key"] == "rh1_primary"
    assert "roc_auc" in payload["criteria"][0]["question"]
    # The harm-policy agreement check is what makes a flipped verdict visible rather than
    # an amendment written afterwards.
    assert any(c["key"] == "harm_policy_agreement" for c in payload["criteria"])


def test_the_verdict_is_read_from_the_donor_marginalized_row(
    cli: dict[str, object],
) -> None:
    """Counting an engine on its most favourable donor would put back the arbitrary choice
    that averaging over donors exists to remove."""
    payload = json.loads((Path(str(cli["results"])) / "rh1_gate.json").read_text(encoding="utf-8"))
    assert any("marginalizes over 3 donor substitutions" in note for note in payload["notes"])


def test_the_recovery_tables_name_the_artifacts_they_came_from(
    cli: dict[str, object],
) -> None:
    """A number that cannot be traced to a saved artifact does not belong in the docs."""
    results = Path(str(cli["results"]))
    sources = json.loads((results / "recovery_tables_sources.json").read_text(encoding="utf-8"))
    assert {"R1", "R2", "R3"} <= set(sources)
    for key in ("R1", "R2", "R3"):
        assert Path(sources[key]).is_file()
    for name in (
        "R1_generator_statistics.csv",
        "R2_h2_readiness_statistics.csv",
        "R3_matched_transfer.csv",
    ):
        assert (results / name).is_file()


def test_the_readiness_table_states_which_risk_targets_are_reachable_at_all(
    cli: dict[str, object],
) -> None:
    """The point of R2: a target needing more accepted edits than the pool can supply is
    unreachable by any method, which is arithmetic rather than a finding about one."""
    table = pd.read_csv(Path(str(cli["results"])) / "R2_h2_readiness_statistics.csv")
    assert {
        "epsilon",
        "n_accepted_available_oracle",
        "n_accepted_required_if_clean",
        "resolvable",
    } <= set(table.columns)
    tightest = table[table["epsilon"] == table["epsilon"].min()]
    assert not tightest["resolvable"].any(), (
        "the tightest target must be unreachable on this corpus, or the fixture is wrong"
    )
