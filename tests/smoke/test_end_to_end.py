"""The mandatory synthetic end-to-end test.

Proves the *research workflow* composes, not merely that the code imports. It runs the
whole pipeline on a small generated corpus with simulated engines: no network, no GPU, no
dataset, no OCR binary.

What it actually asserts is the set of properties a reviewer would check by hand — every
stage produced an artifact, the provenance chain resolves, nothing leaked, the outputs are
labelled synthetic, and the gate can return each of its verdicts including the negative one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from ocr_risk.analysis.report import build_report
from ocr_risk.analysis.tables import AnalysisInput, LeakyRunError
from ocr_risk.config import load_config
from ocr_risk.experiments import run_experiment
from ocr_risk.experiments.gate import (
    GO,
    H1_INCONCLUSIVE,
    H1_NOT_SUPPORTED,
    H1_PARTIALLY_SUPPORTED,
    H1_SUPPORTED,
    INCONCLUSIVE,
    NO_GO,
    evaluate_gate,
)
from ocr_risk.io.artifacts import ArtifactStore
from ocr_risk.io.paths import project_root
from ocr_risk.provenance import build_lineage
from ocr_risk.schemas.enums import StageName
from ocr_risk.splits import audit_records

PIPELINE_STAGES = (
    StageName.MANIFEST,
    StageName.CANONICALIZE,
    StageName.ALIGN,
    StageName.SITES,
    StageName.CANDIDATES,
    StageName.PREDICT,
    StageName.DECIDE,
)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, object]]:
    """Run the whole pipeline once into an isolated tree, and report on it.

    The environment is restored on teardown. A module-scoped fixture that mutates
    os.environ without restoring it leaks into every test that runs afterwards — here it
    redirected the real-corpus tests at an empty temporary tree, so they failed in the
    full suite and passed in isolation.
    """
    root = tmp_path_factory.mktemp("smoke")
    import os

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

    resolved = load_config(
        project_root() / "configs/experiments/smoke_synthetic.yaml",
        # A smaller corpus keeps the test near a minute while exercising every stage.
        ["datasets=[{id: synthetic, enabled: true, params: {n_documents: 16, seed: 4242}}]"],
    )
    store = ArtifactStore()
    state = run_experiment(resolved, store)
    report = build_report(resolved, root / "results", store)
    try:
        yield {
            "root": root,
            "resolved": resolved,
            "store": store,
            "state": state,
            "report": report,
        }
    finally:
        for var, previous in saved.items():
            if previous is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = previous


# --- every stage ran ------------------------------------------------------------------
def test_every_pipeline_stage_produced_an_artifact(smoke: dict[str, object]) -> None:
    store: ArtifactStore = smoke["store"]  # type: ignore[assignment]
    for stage in PIPELINE_STAGES:
        record = store.latest(stage, "smoke_synthetic")
        assert record is not None, f"stage {stage.value} produced no run"
        assert record.outputs, f"stage {stage.value} produced no outputs"


def test_output_hashes_still_match(smoke: dict[str, object]) -> None:
    """Every artifact must hash to what its run record claims."""
    store: ArtifactStore = smoke["store"]  # type: ignore[assignment]
    for stage in PIPELINE_STAGES:
        record = store.latest(stage, "smoke_synthetic")
        assert record is not None
        assert store.verify_outputs(record), f"{stage.value} outputs no longer match their hashes"


def test_the_pipeline_produced_real_data(smoke: dict[str, object]) -> None:
    state = smoke["state"]
    assert len(state.documents) == 16  # type: ignore[attr-defined]
    assert state.spans  # type: ignore[attr-defined]
    assert state.sites  # type: ignore[attr-defined]
    assert state.candidates  # type: ignore[attr-defined]
    assert state.labels  # type: ignore[attr-defined]
    assert state.evidence  # type: ignore[attr-defined]
    assert state.fold_results  # type: ignore[attr-defined]


def test_all_four_engines_read_every_document(smoke: dict[str, object]) -> None:
    """Matched-source is the benchmark's core principle: engine shift must not be
    confounded with document shift."""
    state = smoke["state"]
    by_engine: dict[str, set[str]] = {}
    for span in state.spans:  # type: ignore[attr-defined]
        by_engine.setdefault(span.engine_id, set()).add(span.document_id)
    assert len(by_engine) == 4
    assert len({frozenset(v) for v in by_engine.values()}) == 1


# --- provenance ---------------------------------------------------------------------------
def test_provenance_walks_back_to_the_source_manifest(smoke: dict[str, object]) -> None:
    store: ArtifactStore = smoke["store"]  # type: ignore[assignment]
    decide = store.latest(StageName.DECIDE, "smoke_synthetic")
    assert decide is not None

    lineage = build_lineage(store, decide.run_id)
    assert lineage.complete, (
        f"broken provenance chain: missing={lineage.missing_runs} "
        f"corrupted={lineage.corrupted_runs}"
    )
    stages = {node.record.stage for node in lineage.nodes}
    assert StageName.MANIFEST in stages, "the chain does not reach the source manifest"
    assert stages >= set(PIPELINE_STAGES)


def test_synthetic_taint_propagates_the_whole_way(smoke: dict[str, object]) -> None:
    """A downstream stage must not be able to launder a synthetic input into a clean output."""
    store: ArtifactStore = smoke["store"]  # type: ignore[assignment]
    for stage in PIPELINE_STAGES:
        record = store.latest(stage, "smoke_synthetic")
        assert record is not None
        assert record.synthetic, f"{stage.value} lost the synthetic flag"


def test_run_records_capture_reproduction_metadata(smoke: dict[str, object]) -> None:
    """Given only the record, could someone re-derive this?"""
    store: ArtifactStore = smoke["store"]  # type: ignore[assignment]
    record = store.latest(StageName.DECIDE, "smoke_synthetic")
    assert record is not None
    assert record.git.commit
    assert record.code_sha256 and record.code_sha256 != "unknown"
    assert record.config_sha256
    assert record.environment.package_digest
    assert record.outputs_digest


# --- leakage ------------------------------------------------------------------------------
def test_leakage_audit_passes_on_the_smoke_run(smoke: dict[str, object]) -> None:
    store: ArtifactStore = smoke["store"]  # type: ignore[assignment]
    audit = audit_records(list(store.iter_records()))
    assert audit.passed, [str(f) for f in audit.findings]
    assert audit.folds_checked > 0


def test_the_document_partition_is_shared_by_every_fold(smoke: dict[str, object]) -> None:
    """A partition redrawn per fold would rotate documents between roles and silently undo
    the document split."""
    state = smoke["state"]
    hashes = {r.descriptor.document_partition_sha256 for r in state.fold_results}  # type: ignore[attr-defined]
    assert len(hashes) == 1


def test_no_fold_evaluates_an_engine_it_was_fitted_on(smoke: dict[str, object]) -> None:
    state = smoke["state"]
    for result in state.fold_results:  # type: ignore[attr-defined]
        split = result.descriptor
        assert not (set(split.fit_engines) & set(split.evaluate_engines))
        assert not (set(split.calibrate_engines) & set(split.evaluate_engines))


# --- analysis outputs -----------------------------------------------------------------------
def test_report_produces_tables_and_figures(smoke: dict[str, object]) -> None:
    report = smoke["report"]
    assert report.tables  # type: ignore[attr-defined]
    assert report.figures  # type: ignore[attr-defined]
    for path in list(report.tables.values()) + list(report.figures.values()):  # type: ignore[attr-defined]
        assert path.is_file(), f"{path} was not written"
        assert path.stat().st_size > 0


def test_harm_policy_sensitivity_is_reported(smoke: dict[str, object]) -> None:
    """The harm definition moves the headline numbers, so all three must be reported."""
    report = smoke["report"]
    names = set(report.tables)  # type: ignore[attr-defined]
    assert "coverage_at_risk" in names
    assert "coverage_at_risk__non_improving" in names
    assert "coverage_at_risk__exact_only" in names


def test_the_challenge_set_is_reported_separately(smoke: dict[str, object]) -> None:
    """Adversarial hard negatives must not be pooled into the headline: doing so measures
    adversarial rejection rather than safe repair automation."""
    report = smoke["report"]
    assert "challenge_set_hard_negatives" in report.tables  # type: ignore[attr-defined]
    challenge = pd.read_csv(report.tables["challenge_set_hard_negatives"])  # type: ignore[index]
    headline = pd.read_csv(report.tables["risk_coverage_summary"])  # type: ignore[index]
    if not challenge.empty and not headline.empty:
        # The challenge set is adversarial by construction, so its base harm rate must be
        # visibly higher than the natural pool's.
        assert challenge["base_harm_rate"].mean() > headline["base_harm_rate"].mean()


def test_every_output_is_labelled_synthetic(smoke: dict[str, object]) -> None:
    report = smoke["report"]
    manifest = json.loads(report.manifest_path.read_text())  # type: ignore[attr-defined]
    assert manifest["synthetic"] is True
    assert "NOT A RESEARCH RESULT" in manifest["banner"]

    for path in report.tables.values():  # type: ignore[attr-defined]
        frame = pd.read_csv(path)
        if not frame.empty:
            assert "synthetic" in frame.columns
            assert bool(frame["synthetic"].all())


def test_figure_manifest_records_content_hashes(smoke: dict[str, object]) -> None:
    """Traceability: every figure must name the runs and hashes it came from."""
    report = smoke["report"]
    manifest = json.loads(report.manifest_path.read_text())  # type: ignore[attr-defined]
    assert manifest["source_runs"]
    assert manifest["config_sha256"]
    output_dir: Path = report.output_dir  # type: ignore[attr-defined]

    from ocr_risk.io.hashing import file_sha256

    for filename, entry in manifest["outputs"].items():
        path = output_dir / filename
        assert path.is_file(), f"{filename} listed in the manifest but missing on disk"
        assert file_sha256(path) == entry["sha256"], f"{filename} does not match its hash"

    # Every table and figure on disk must have an entry. The manifest was keyed by LOGICAL
    # name and built by merging the table and figure dicts, so any output whose CSV and PNG
    # shared a name lost its CSV entry -- coverage_at_risk, calibration and
    # confidence_reliability among them. Traceability was inverted: the primary-harm-policy
    # tables had no hash while their sensitivity variants did.
    produced = {p.name for p in output_dir.iterdir() if p.suffix in {".csv", ".png"}}
    recorded = set(manifest["outputs"])
    assert produced <= recorded, f"missing manifest entries: {sorted(produced - recorded)}"


# --- the gate -----------------------------------------------------------------------------------
def test_gate_emits_a_verdict_for_every_hypothesis(smoke: dict[str, object]) -> None:
    report = smoke["report"]
    resolved = smoke["resolved"]
    gate = evaluate_gate(
        experiment="smoke_synthetic",
        synthetic=True,
        coverage_table=pd.read_csv(report.tables["coverage_at_risk"]),  # type: ignore[index]
        calibration=pd.read_csv(report.tables["calibration"]),  # type: ignore[index]
        harm=pd.read_csv(report.tables["harm_decomposition"]),  # type: ignore[index]
        epsilon=max(resolved.config.risk.epsilon_grid),  # type: ignore[attr-defined]
    )
    assert {v.hypothesis for v in gate.verdicts} == {"H1", "H2", "H3", "H4"}
    h1_verdicts = {H1_SUPPORTED, H1_PARTIALLY_SUPPORTED, H1_NOT_SUPPORTED, H1_INCONCLUSIVE}
    assert all(
        v.verdict in (h1_verdicts if v.hypothesis == "H1" else {GO, NO_GO, INCONCLUSIVE})
        for v in gate.verdicts
    )
    assert all(v.implication for v in gate.verdicts)
    assert any("SYNTHETIC" in note for note in gate.notes)


def test_gate_can_return_no_go() -> None:
    """The negative branch must be reachable, or a NO_GO could never be reported.

    Constructed so V6 loses to V3 and overcorrection exceeds the tolerance: the two
    failures H2 and H4 are meant to detect.
    """
    coverage = pd.DataFrame(
        [
            {"fold_id": f"f{i}", "verifier_id": v, "epsilon": 0.1, "coverage": c}
            for i, (v, c) in enumerate(
                [
                    ("v3_text_conf", 0.40),
                    ("v6_full", 0.10),
                    ("v3_text_conf", 0.42),
                    ("v6_full", 0.11),
                ]
            )
        ]
    )
    coverage["fold_id"] = ["fold_a", "fold_a", "fold_b", "fold_b"]
    calibration = pd.DataFrame(
        [{"fold_id": "fold_a", "brier": 0.20}, {"fold_id": "fold_b", "brier": 0.201}]
    )
    harm = pd.DataFrame(
        [
            {
                "n_accepted": 50,
                "overcorrection_rate": 0.9,
                "accepted_edit_risk": 0.9,
                "epsilon": 0.1,
            }
        ]
    )

    gate = evaluate_gate(
        experiment="negative_branch",
        synthetic=False,
        coverage_table=coverage,
        calibration=calibration,
        harm=harm,
        epsilon=0.1,
    )
    # H1 is a contrast between two protocols. A run that supplies neither must say it
    # could not tell, not that the effect is absent.
    paired = pd.DataFrame(
        [
            {
                "fold_id": "fold_a",
                "epsilon": 0.1,
                "delta": -0.30,
                "ci_lower": -0.4,
                "ci_upper": -0.2,
            },
            {
                "fold_id": "fold_b",
                "epsilon": 0.1,
                "delta": -0.31,
                "ci_lower": -0.4,
                "ci_upper": -0.2,
            },
        ]
    )
    transfer = pd.DataFrame(
        [
            {
                "held_out_engine": engine,
                "verifier_id": "v6_full",
                "metric": "calibration_error",
                "delta": 0.0001,
                "degraded_after_holm": False,
            }
            for engine in ("engine_a", "engine_b", "engine_c", "engine_d")
        ]
    )
    gate = evaluate_gate(
        experiment="negative_branch",
        synthetic=False,
        coverage_table=coverage,
        calibration=calibration,
        harm=harm,
        epsilon=0.1,
        transfer=transfer,
        paired_h2=paired,
    )
    verdicts = {v.hypothesis: v.verdict for v in gate.verdicts}
    assert verdicts["H1"] == H1_NOT_SUPPORTED, (
        "calibration transferred evenly to every engine but H1 did not report NOT SUPPORTED"
    )
    assert verdicts["H2"] == NO_GO, "pixels lost to text+confidence but H2 did not report NO_GO"
    assert verdicts["H4"] == NO_GO, "risk exceeded tolerance but H4 did not report NO_GO"


def test_gate_distinguishes_inconclusive_from_no_go() -> None:
    """ "We looked and it did not help" and "we could not tell" call for different next
    steps; collapsing them would let a weak experiment look like evidence of absence."""
    gate = evaluate_gate(
        experiment="empty",
        synthetic=False,
        coverage_table=pd.DataFrame(),
        calibration=pd.DataFrame(),
        harm=pd.DataFrame(),
        epsilon=0.1,
    )
    assert all(
        v.verdict == (H1_INCONCLUSIVE if v.hypothesis == "H1" else INCONCLUSIVE)
        for v in gate.verdicts
    )


# --- headline guard ---------------------------------------------------------------------------------
def test_a_leaky_fold_cannot_enter_a_headline_table() -> None:
    """The doc-overlap diagnostic exists to measure inflation, not to be reported."""
    data = AnalysisInput(
        predictions=pd.DataFrame(),
        decisions=pd.DataFrame(),
        labels=pd.DataFrame(),
        candidates=pd.DataFrame(),
        leaky_folds=frozenset({"diagnostic_doc_overlap:synth_a"}),
    )
    with pytest.raises(LeakyRunError, match="contaminated by design"):
        data.guard_headline({"diagnostic_doc_overlap:synth_a"})


@pytest.mark.parametrize(
    "table_fn",
    ["coverage_at_risk_table", "risk_coverage_table", "calibration_table"],
)
def test_every_headline_table_refuses_a_leaky_fold(table_fn: str) -> None:
    """Not just one of them.

    The guard was originally applied only to risk_coverage_summary, leaving
    Coverage@Risk — the primary result, and the number most likely to be quoted —
    unprotected.
    """
    from ocr_risk.analysis import tables as tables_module
    from ocr_risk.schemas.enums import HarmPolicy

    leaky_fold = "diagnostic_doc_overlap:synth_a"
    predictions = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "site_id": "s1",
                "document_id": "d1",
                "dataset_id": "ds",
                "engine_id": "e",
                "verifier_id": "v6_full",
                "evidence_config": "v6",
                "fold_id": leaky_fold,
                "raw_score": 0.8,
                "calibrated_score": 0.8,
            }
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "outcome_if_accepted": "true_correction",
                "d_before": 1,
                "delta": 1,
            }
        ]
    )
    candidates = pd.DataFrame([{"candidate_id": "c1", "is_synthetic_hard_negative": False}])

    data = tables_module.AnalysisInput(
        predictions=predictions,
        decisions=pd.DataFrame(),
        labels=labels,
        candidates=candidates,
        leaky_folds=frozenset({leaky_fold}),
    )

    function = getattr(tables_module, table_fn)
    args = (
        (data, HarmPolicy.STRICT_WORSENING, (0.1,))
        if table_fn == "coverage_at_risk_table"
        else (data, HarmPolicy.STRICT_WORSENING)
    )
    with pytest.raises(LeakyRunError, match="contaminated by design"):
        function(*args)


def test_the_gate_report_names_the_harm_policy_it_judged() -> None:
    """The H1 verdict is policy-dependent, so a report that does not name its policy
    cannot be interpreted.

    On the real pilot the same candidates give 0/4 degraded engines under
    ``strict_worsening`` and 2/4 under ``non_improving`` -- NOT SUPPORTED versus
    PARTIALLY SUPPORTED off one config field. A reader handed the verdict without the
    policy has been handed a number they cannot check.
    """
    transfer = pd.DataFrame(
        [
            {
                "held_out_engine": engine,
                "verifier_id": "v6_full",
                "metric": "calibration_error",
                "delta": 0.0001,
                "degraded_after_holm": False,
            }
            for engine in ("engine_a", "engine_b", "engine_c", "engine_d")
        ]
    )
    gate = evaluate_gate(
        experiment="policy_stamp",
        synthetic=False,
        coverage_table=pd.DataFrame(),
        calibration=pd.DataFrame(),
        harm=pd.DataFrame(),
        epsilon=0.1,
        transfer=transfer,
        harm_policy="non_improving",
    )
    assert gate.as_dict()["harm_policy"] == "non_improving", (
        "the gate report did not record which harm policy produced the verdict"
    )
