"""Assembling the analysis report from artifacts.

Reads the decide-stage run and everything upstream of it, produces tables and figures, and
writes a ``figure_manifest.json`` recording the run ids and content hashes each output was
derived from. That manifest is what makes ``ocr-risk provenance`` able to answer "where did
this figure come from?" without anyone remembering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ocr_risk.analysis.figures import (
    FigureSpec,
    plot_confidence_reliability,
    plot_coverage_at_risk,
    plot_reliability,
    plot_risk_coverage,
)
from ocr_risk.analysis.tables import (
    AnalysisInput,
    calibration_table,
    candidate_population,
    confidence_reliability,
    coverage_at_risk_table,
    deployed_operating_table,
    engine_diagnostics,
    harm_decomposition,
    noise_tertiles,
    risk_coverage_table,
    stratified_risk,
    transfer_matrix,
)
from ocr_risk.config import ResolvedConfig
from ocr_risk.io.artifacts import ArtifactStore
from ocr_risk.io.hashing import file_sha256
from ocr_risk.metrics import RiskCoverageCurve, cer, risk_coverage_curve
from ocr_risk.schemas.enums import HarmPolicy, StageName, harmful_outcomes

__all__ = ["ReportResult", "build_report"]

_SYNTHETIC_BANNER = (
    "SYNTHETIC - NOT A RESEARCH RESULT. Produced from generated documents and simulated "
    "OCR engines to validate the pipeline architecture. These numbers say nothing about "
    "real OCR and must never be quoted as findings."
)


@dataclass(slots=True)
class ReportResult:
    """Everything the report stage produced."""

    output_dir: Path
    tables: dict[str, Path]
    figures: dict[str, Path]
    manifest_path: Path
    synthetic: bool


def _load(
    store: ArtifactStore, experiment: str, pool_experiment: str | None = None
) -> tuple[AnalysisInput, dict[str, str]]:
    """Read the artifact tables this layer is allowed to see.

    ``pool_experiment`` names where the candidates and sites came from, for a study that
    reuses another experiment's pool rather than generating one. RH1 does exactly that, so
    that the only difference from the H1 pilot is the design; without this the analysis
    layer would look for a candidates run under the RH1 name and find none.
    """
    upstream = pool_experiment or experiment
    decide_run = store.latest(StageName.DECIDE, experiment)
    predict_run = store.latest(StageName.PREDICT, experiment)
    candidates_run = store.latest(StageName.CANDIDATES, upstream)
    sites_run = store.latest(StageName.SITES, upstream)
    missing = [
        name
        for name, record in (
            ("decide", decide_run),
            ("predict", predict_run),
            ("candidates", candidates_run),
            ("sites", sites_run),
        )
        if record is None
    ]
    if missing:
        msg = f"experiment {experiment!r} is missing runs: {', '.join(missing)}"
        raise FileNotFoundError(msg)
    assert decide_run and predict_run and candidates_run and sites_run

    folds = store.read_json(predict_run, "folds.json")
    leaky = frozenset(
        f["fold_id"] for f in folds if f.get("split", {}).get("leaky") or f.get("leaky")
    )

    data = AnalysisInput(
        predictions=store.read_table(predict_run, "predictions").to_pandas(),
        decisions=store.read_table(decide_run, "decisions").to_pandas(),
        labels=store.read_table(candidates_run, "labels").to_pandas(),
        candidates=store.read_table(candidates_run, "candidates").to_pandas(),
        sites=store.read_table(sites_run, "sites").to_pandas(),
        leaky_folds=leaky,
    )
    sources = {
        "decide": decide_run.run_id,
        "predict": predict_run.run_id,
        "candidates": candidates_run.run_id,
        "sites": sites_run.run_id,
    }
    return data, sources


def build_report(
    resolved: ResolvedConfig,
    output_dir: Path,
    store: ArtifactStore | None = None,
) -> ReportResult:
    """Produce every table and figure for one experiment."""
    store = store or ArtifactStore()
    cfg = resolved.config
    data, sources = _load(store, cfg.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic = cfg.synthetic
    tables: dict[str, Path] = {}
    figures: dict[str, Path] = {}

    def write_table(name: str, frame: pd.DataFrame) -> None:
        path = output_dir / f"{name}.csv"
        annotated = frame.copy()
        if not annotated.empty:
            annotated.insert(0, "synthetic", synthetic)
        annotated.to_csv(path, index=False)
        tables[name] = path

    # --- the headline tables, and the sensitivity analysis over harm policy -------------
    primary = cfg.risk.harm_policy
    for policy in cfg.risk.sensitivity_harm_policies:
        suffix = "" if policy is primary else f"__{policy.value}"
        write_table(
            f"coverage_at_risk{suffix}",
            coverage_at_risk_table(
                data,
                policy,
                cfg.risk.epsilon_grid,
                n_bootstrap=cfg.stats.n_bootstrap,
                ci_level=cfg.stats.ci_level,
                seed=cfg.stats.bootstrap_seed,
            ),
        )
        write_table(f"risk_coverage_summary{suffix}", risk_coverage_table(data, policy))
        write_table(f"harm_decomposition{suffix}", harm_decomposition(data, policy))

    # The deployable measurement: the threshold the controller certified on calibration,
    # applied unchanged. Distinct from coverage_at_risk, which is an in-sample optimum and
    # therefore satisfies its own tolerance by construction.
    decide_record = store.latest(StageName.DECIDE, cfg.name)
    assert decide_record is not None
    thresholds = pd.DataFrame(store.read_json(decide_record, "thresholds.json"))
    for policy in cfg.risk.sensitivity_harm_policies:
        suffix = "" if policy is primary else f"__{policy.value}"
        write_table(
            f"deployed_operating_points{suffix}",
            deployed_operating_table(
                data,
                policy,
                thresholds,
                n_bootstrap=cfg.stats.n_bootstrap,
                ci_level=cfg.stats.ci_level,
                seed=cfg.stats.bootstrap_seed,
            ),
        )

    write_table("calibration", calibration_table(data, primary, n_bins=cfg.calibration.n_bins))
    # Per-engine confounds, in the same output as the risk numbers. Alignment ambiguity,
    # site counts and span granularity all differ by engine for reasons unrelated to
    # recognition, and any of them could be mistaken for a transfer effect if it were
    # only available in a separate diagnostics file nobody reads beside the result.
    write_table("engine_diagnostics", engine_diagnostics(store, cfg.name))
    write_table("candidate_population", candidate_population(data, primary))
    reliability_by_confidence = confidence_reliability(store, cfg.name)
    write_table("confidence_reliability", reliability_by_confidence)

    # --- pre-registered stratified analyses ---------------------------------------------
    manifest_record = store.latest(StageName.MANIFEST, cfg.name)
    if manifest_record is not None:
        documents = store.read_table(manifest_record, "documents").to_pandas()
        by_corpus = dict(zip(documents["document_id"], documents["dataset_id"], strict=True))
        write_table(
            "stratified_by_corpus",
            stratified_risk(data, primary, by_corpus, "corpus"),
        )
        write_table(
            "stratified_by_noise",
            stratified_risk(data, primary, _noise_stratum(store, cfg.name, documents), "noise"),
        )
    # primary_epsilon, not max(grid). Taking the maximum is the exact selection rule the
    # pre-registration forbids -- the most permissive tolerance is the one most
    # favourable to a positive number. Harmless while they coincide, silent when they
    # stop coinciding.
    write_table("transfer_matrix", transfer_matrix(data, primary, cfg.risk.primary_epsilon))
    # The adversarial challenge set, reported separately and never pooled into the above.
    write_table(
        "challenge_set_hard_negatives",
        risk_coverage_table(data, primary, pool="challenge"),
    )

    # --- figures ------------------------------------------------------------------------
    caption_base = (
        f"{cfg.name} | harm policy {primary.value} | "
        f"protocol {cfg.splits.protocol} | natural candidate pool "
        "(adversarial hard negatives reported separately)"
    )
    source_runs = tuple(sources.values())

    curves = _curves_for_figure(data, primary)
    figures["risk_coverage"] = plot_risk_coverage(
        curves,
        FigureSpec(
            path=output_dir / "risk_coverage.png",
            title="Risk-coverage frontier by evidence configuration",
            caption=caption_base,
            synthetic=synthetic,
            source_runs=source_runs,
            config_sha256=resolved.sha256,
        ),
        epsilons=cfg.risk.epsilon_grid,
    )
    figures["confidence_reliability"] = plot_confidence_reliability(
        reliability_by_confidence,
        FigureSpec(
            path=output_dir / "confidence_reliability.png",
            title="Native OCR confidence vs empirical correctness, per engine",
            caption=(
                f"{cfg.name} | descriptive, computed before any verifier | resolved "
                "alignment components only"
            ),
            synthetic=synthetic,
            source_runs=source_runs,
            config_sha256=resolved.sha256,
        ),
    )
    figures["coverage_at_risk"] = plot_coverage_at_risk(
        pd.read_csv(tables["coverage_at_risk"]),
        FigureSpec(
            path=output_dir / "coverage_at_risk.png",
            title="Coverage at each harmful-edit tolerance (document-level 95% CI)",
            caption=caption_base,
            synthetic=synthetic,
            source_runs=source_runs,
            config_sha256=resolved.sha256,
        ),
    )
    figures["calibration"] = plot_reliability(
        pd.read_csv(tables["calibration"]),
        FigureSpec(
            path=output_dir / "calibration.png",
            title="Calibration on the held-out engine (Brier primary, ECE secondary)",
            caption=caption_base,
            synthetic=synthetic,
            source_runs=source_runs,
            config_sha256=resolved.sha256,
        ),
    )

    # --- the manifest that makes all of it traceable ---------------------------------------
    manifest_path = output_dir / "figure_manifest.json"
    manifest = {
        "experiment": cfg.name,
        "config_sha256": resolved.sha256,
        "synthetic": synthetic,
        "banner": _SYNTHETIC_BANNER if synthetic else "",
        "harm_policy": primary.value,
        "sensitivity_harm_policies": [p.value for p in cfg.risk.sensitivity_harm_policies],
        "source_run_id": sources["decide"],
        "source_runs": sources,
        # Keyed by FILENAME, not by logical name. Merging the two dicts silently dropped
        # every output whose table and figure share a name -- coverage_at_risk, calibration
        # and confidence_reliability -- so the primary-harm-policy CSVs had no hash while
        # their sensitivity variants did. Traceability was inverted with respect to which
        # number is the headline.
        "outputs": {
            path.name: {"kind": kind, "name": name, "sha256": file_sha256(path)}
            for kind, group in (("table", tables), ("figure", figures))
            for name, path in sorted(group.items())
        },
        # Written by `analyze h1` and `gate pilot` rather than by this function, so they
        # are named here as expected companions. A verdict resting on a file with no
        # provenance entry is the one gap this manifest exists to close.
        "companion_outputs": [
            "h1_transfer.csv",
            "h1_transfer_brier.png",
            "h1_transfer_calibration_error.png",
            "h1_transfer_refinement_error.png",
            "h1_transfer_ece_equal_mass.png",
            "gate_report.json",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")

    if synthetic:
        (output_dir / "README.md").write_text(
            f"# {cfg.name}\n\n> **{_SYNTHETIC_BANNER}**\n\n"
            f"Config hash `{resolved.sha256}`. Regenerate with:\n\n"
            f"```bash\nmake smoke\n```\n\n"
            f"Trace any output back to its source artifacts:\n\n"
            f"```bash\nocr-risk provenance {manifest_path.name}\n```\n",
            encoding="utf-8",
        )

    return ReportResult(
        output_dir=output_dir,
        tables=tables,
        figures=figures,
        manifest_path=manifest_path,
        synthetic=synthetic,
    )


def _curves_for_figure(data: AnalysisInput, policy: HarmPolicy) -> dict[str, RiskCoverageCurve]:
    """One curve per (fold, verifier), over the natural candidate pool.

    Per fold, never pooled. The document partition is global and identical across folds,
    so pooling would count every test document once per engine and mix four different
    held-out-engine distributions into a single curve — hiding precisely the between-
    engine variation the study is about, and narrowing it by a factor of the fold count.
    """
    frame = data.predictions.merge(
        data.labels[["candidate_id", "outcome_if_accepted"]], on="candidate_id", how="inner"
    ).merge(
        data.candidates[["candidate_id", "is_synthetic_hard_negative"]],
        on="candidate_id",
        how="left",
    )
    frame = data.natural(frame)
    if not frame.empty:
        data.guard_headline(set(frame["fold_id"].unique()))
    harmful_values = {o.value for o in harmful_outcomes(policy)}

    curves: dict[str, RiskCoverageCurve] = {}
    for (fold_id, verifier_id), group in frame.groupby(["fold_id", "verifier_id"], sort=True):
        scores = group["calibrated_score"].fillna(group["raw_score"]).to_numpy(dtype=float)
        harmful = group["outcome_if_accepted"].isin(harmful_values).to_numpy(dtype=bool)
        if scores.size:
            curves[f"{fold_id} | {verifier_id}"] = risk_coverage_curve(scores, harmful)
    return curves


def _noise_stratum(
    store: ArtifactStore, experiment: str, documents: pd.DataFrame
) -> dict[str, str]:
    """Per-document OCR noise tertile, computed within each (corpus, engine) cell.

    Collapsed to one label per document by majority across engines, because the analysis
    frame is keyed by document. A document whose engines disagree about its difficulty is
    genuinely ambiguous, and taking the modal label is the least assuming resolution.
    """
    canonical = store.latest(StageName.CANONICALIZE, experiment)
    if canonical is None:
        return {}
    spans = store.read_table(canonical, "spans").to_pandas()
    gt_of = dict(zip(documents["document_id"], documents["gt_text"], strict=True))

    rates: dict[tuple[str, str], float] = {}
    for (document_id, engine_id), group in spans.groupby(["document_id", "engine_id"]):
        reference = gt_of.get(document_id)
        if not reference:
            continue
        hypothesis = " ".join(group.sort_values("reading_order")["text"].astype(str))
        rates[(str(document_id), str(engine_id))] = cer(reference, hypothesis)

    tertiles = noise_tertiles(documents, rates)
    per_document: dict[str, list[str]] = {}
    for (document_id, _), label in tertiles.items():
        per_document.setdefault(document_id, []).append(label)
    return {
        document_id: max(set(labels), key=labels.count)
        for document_id, labels in per_document.items()
    }
