"""``ocr-risk analyze`` and ``ocr-risk gate`` — reporting and the pilot verdict."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ocr_risk.analysis.figures import FigureSpec, plot_transfer_gap
from ocr_risk.analysis.report import _load, build_report
from ocr_risk.analysis.transfer import h1_transfer_table
from ocr_risk.config import ResolvedConfig, load_config
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
from ocr_risk.io.hashing import file_sha256
from ocr_risk.io.paths import project_root
from ocr_risk.schemas.enums import HarmPolicy, StageName

app = typer.Typer(name="analyze", help="Generate tables and figures.", no_args_is_help=True)
gate_app = typer.Typer(name="gate", help="Pilot go/no-go verdict.", no_args_is_help=True)
console = Console()


def _experiment_configs() -> dict[str, Path]:
    """Map experiment name to config, discovered rather than listed.

    A hardcoded list drifts: it named a `loeo_zero_shot.yaml` that does not exist and
    omitted every config added since. The name comes from the resolved config rather than
    the filename, because the two are allowed to differ.
    """
    directory = project_root() / "configs" / "experiments"
    found: dict[str, Path] = {}
    for path in sorted(directory.glob("*.yaml")):
        try:
            found[load_config(path).config.name] = path
        except Exception:
            continue
    return found


def _resolve_config(experiment: str, config_path: Path | None) -> ResolvedConfig:
    if config_path is not None:
        return load_config(config_path)
    known = _experiment_configs()
    if experiment not in known:
        raise typer.BadParameter(
            f"unknown experiment {experiment!r}; pass --config explicitly "
            f"(known: {', '.join(sorted(known))})"
        )
    return load_config(known[experiment])


@app.command("report")
def report(
    experiment: Annotated[str, typer.Option("--experiment", "-e")],
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path(
        "results/generated"
    ),
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Produce every table and figure, plus the manifest that makes them traceable."""
    resolved = _resolve_config(experiment, config_path)
    result = build_report(resolved, out)

    console.print(f"[green]report[/green] -> {result.output_dir}")
    for name, path in sorted(result.tables.items()):
        rows = len(pd.read_csv(path)) if path.stat().st_size else 0
        console.print(f"  table  {name:38} {rows:>5} rows")
    for name, path in sorted(result.figures.items()):
        console.print(f"  figure {name:38} {path.name}")
    console.print(f"  manifest {result.manifest_path.name}")

    if result.synthetic:
        console.print(
            "[yellow]SYNTHETIC — every output is watermarked and must not be quoted as a "
            "research result[/yellow]"
        )


@gate_app.command("pilot")
def pilot(
    experiment: Annotated[str, typer.Option("--experiment", "-e")],
    results: Annotated[Path, typer.Option("--results")] = Path("results/generated"),
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    harm_policy: Annotated[HarmPolicy | None, typer.Option("--harm-policy")] = None,
) -> None:
    """Judge H1-H4 against the pre-registered criteria.

    Always emits a verdict. NO_GO is a real, reportable outcome, and INCONCLUSIVE is kept
    distinct from it: "we looked and it did not help" and "we could not tell" call for
    different next steps.
    """
    resolved = _resolve_config(experiment, config_path)
    cfg = resolved.config
    policy = harm_policy or cfg.risk.harm_policy

    def read(name: str) -> pd.DataFrame:
        path = results / f"{name}.csv"
        if not path.is_file():
            return pd.DataFrame()
        return pd.read_csv(path)

    # `analyze h1` writes the configured policy to the bare name and each other policy to
    # a `__<policy>` suffix, so the sensitivity variants are judgeable by the same gate
    # rather than by a reader eyeballing a CSV.
    transfer_name = "h1_transfer" if policy is cfg.risk.harm_policy else f"h1_transfer__{policy}"
    transfer = read(transfer_name)
    if transfer.empty and policy is not cfg.risk.harm_policy:
        raise typer.BadParameter(
            f"no transfer table for harm policy {policy!s} at {results / (transfer_name + '.csv')}"
        )

    coverage = read("coverage_at_risk")
    if coverage.empty:
        raise typer.BadParameter(
            f"no analysis tables in {results}; run `ocr-risk analyze report` first"
        )

    report_result = evaluate_gate(
        experiment=cfg.name,
        synthetic=cfg.synthetic,
        coverage_table=coverage,
        calibration=read("calibration"),
        harm=read("harm_decomposition"),
        # The PRIMARY tolerance from the pre-registration, not max(grid). Taking the
        # maximum would silently select the most permissive level -- the one most
        # favourable to a positive verdict -- which is a choice, and not one anybody made.
        epsilon=cfg.risk.primary_epsilon,
        transfer=transfer,
        primary_verifier=cfg.risk.primary_verifier,
        paired_h2=read("h2_paired"),
        harm_policy=str(policy),
    )
    name = "gate_report.json" if policy is cfg.risk.harm_policy else f"gate_report__{policy}.json"
    path = report_result.save(results / name)

    table = Table("hypothesis", "verdict", "observed", title="pilot gate")
    styles = {
        GO: "green",
        NO_GO: "red",
        INCONCLUSIVE: "yellow",
        H1_SUPPORTED: "green",
        H1_PARTIALLY_SUPPORTED: "cyan",
        H1_NOT_SUPPORTED: "red",
        H1_INCONCLUSIVE: "yellow",
    }
    for verdict in report_result.verdicts:
        table.add_row(
            f"{verdict.hypothesis}  {verdict.question}",
            f"[{styles[verdict.verdict]}]{verdict.verdict}[/{styles[verdict.verdict]}]",
            verdict.observed,
        )
    console.print(table)
    for verdict in report_result.verdicts:
        console.print(f"  [dim]{verdict.hypothesis}:[/dim] {verdict.implication}")
    for note in report_result.notes:
        console.print(f"[yellow]{note}[/yellow]")
    console.print(f"  wrote {path}")


@app.command("h1")
def h1(
    zero_shot: Annotated[str, typer.Option("--zero-shot", help="Zero-shot experiment name")],
    oracle: Annotated[str, typer.Option("--oracle", help="Engine-aware reference experiment")],
    results: Annotated[Path, typer.Option("--results")] = Path("results/generated"),
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """The H1 measurement: the paired oracle-versus-zero-shot transfer table.

    H1 is a contrast between two protocols on the same target documents, so it cannot be
    read off a single run. Both arms must have been executed; this command joins them on
    matched candidates and writes ``h1_transfer.csv``, which ``gate pilot`` then judges.
    """
    resolved = _resolve_config(zero_shot, config_path)
    cfg = resolved.config
    store = ArtifactStore()

    left, _ = _load(store, zero_shot)
    right, _ = _load(store, oracle)

    def contrast(policy: HarmPolicy) -> pd.DataFrame:
        return h1_transfer_table(
            left,
            right,
            policy,
            n_bins=cfg.calibration.n_bins,
            n_bootstrap=cfg.stats.n_bootstrap,
            ci_level=cfg.stats.ci_level,
            seed=cfg.stats.bootstrap_seed,
        )

    results.mkdir(parents=True, exist_ok=True)
    table = contrast(cfg.risk.harm_policy)
    path = results / "h1_transfer.csv"
    table.to_csv(path, index=False)

    # The harm policy sets which outcomes count as harmful, hence the `safe` label, hence
    # the base rate, hence the whole calibration analysis. Every other headline table ships
    # all three variants; the table the verdict rests on shipped one. A sensitivity analysis
    # that skips the verdict-bearing table is not a sensitivity analysis.
    sensitivity: list[Path] = []
    for policy in cfg.risk.sensitivity_harm_policies:
        if policy is cfg.risk.harm_policy:
            continue
        variant = results / f"h1_transfer__{policy.value}.csv"
        contrast(policy).to_csv(variant, index=False)
        sensitivity.append(variant)

    figures: list[Path] = []
    if not table.empty:
        for metric in ("calibration_error", "brier", "ece_equal_mass"):
            figures.append(
                plot_transfer_gap(
                    table,
                    FigureSpec(
                        path=results / f"h1_transfer_{metric}.png",
                        title=f"H1: unseen-engine transfer gap in {metric}",
                        caption=(
                            f"{zero_shot} minus {oracle} | paired document-level bootstrap, "
                            f"{cfg.stats.n_bootstrap} resamples | Holm across engines within "
                            "the metric | per engine, never averaged"
                        ),
                        synthetic=cfg.synthetic,
                        source_runs=(zero_shot, oracle),
                        config_sha256=resolved.sha256,
                    ),
                    metric,
                )
            )

    if table.empty:
        console.print("[red]no matched candidates between the two arms[/red]")
        raise typer.Exit(1)

    rendered = Table(
        "engine",
        "verifier",
        "metric",
        "zero-shot",
        "oracle",
        "delta",
        "95% CI",
        "degraded",
        title=f"H1 transfer: {zero_shot} minus {oracle} (positive = transfer made it worse)",
    )
    # Every metric, not only the primary one. Filtering the console view to the endpoint
    # the verdict is read from hides the endpoints that disagree with it at exactly the
    # moment a person is looking -- which on this pilot is where the finding is.
    shown = table[table["verifier_id"] == cfg.risk.primary_verifier]
    for _, row in shown.sort_values(["held_out_engine", "metric"]).iterrows():
        rendered.add_row(
            str(row["held_out_engine"]),
            str(row["verifier_id"]),
            str(row["metric"]),
            f"{row['loeo_zero_shot']:.4f}",
            f"{row['in_engine_oracle']:.4f}",
            f"{row['delta']:+.4f}",
            f"[{row['delta_ci_lower']:+.4f}, {row['delta_ci_upper']:+.4f}]",
            "[red]yes[/red]" if row["degraded_after_holm"] else "no",
        )
    # Record the H1 outputs in the report manifest. The verdict rests on this table, and
    # a load-bearing artifact with no hash entry is exactly what the traceability rule is
    # for -- these are written outside build_report, so they have to register themselves.
    manifest_path = results / "figure_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = manifest.setdefault("outputs", {})
        for produced in [path, *sensitivity, *figures]:
            outputs[produced.name] = {
                "kind": "figure" if produced.suffix == ".png" else "table",
                "name": produced.stem,
                "sha256": file_sha256(produced),
            }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        console.print(f"registered {len(figures) + 1} output(s) in {manifest_path.name}")

    console.print(rendered)
    console.print(f"wrote {path}")
    for figure in figures:
        console.print(f"wrote {figure}")
    console.print(
        "Per target engine, never averaged: four OCR engines are four fixed environments, "
        "not four draws from a population of engines."
    )


@app.command("rh1")
def rh1(
    experiment: Annotated[str, typer.Option("--experiment", "-e")] = "rh1_matched",
    pool: Annotated[
        str, typer.Option("--pool", help="Experiment whose candidate pool the run used")
    ] = "pilot_loeo_zero_shot",
    results: Annotated[Path, typer.Option("--results")] = Path("results/generated"),
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    verifiers: Annotated[
        str | None,
        typer.Option(
            "--verifiers",
            help="Comma-separated verifier ids; default is the "
            "primary plus the two confidence-free controls",
        ),
    ] = None,
) -> None:
    """The RH1 measurement: matched in-engine reference versus LOEO zero-shot.

    Both arms live in one run, so this reads one artifact rather than joining two. The
    primary endpoint is ROC AUC — a pure ranking statistic, invariant to any monotone
    recalibration, and therefore not confounded by the calibration term H1 already
    examined. Every donor substitution is reported; the verdict is read from the pooled
    per-engine row.
    """
    from ocr_risk.analysis.rh1_discrimination import (
        RH1_METRICS,
        RH1_PRIMARY,
        rh1_discrimination_table,
    )

    resolved = _resolve_config(experiment, config_path)
    cfg = resolved.config
    store = ArtifactStore()
    data, sources = _load(store, experiment, pool_experiment=pool)
    # The primary configuration plus the two rungs of the ablation ladder that carry no
    # confidence channel. Those two are the controls for the z-score asymmetry the matched
    # design exists to remove, so they are not optional; the remaining four are context and
    # cost a bootstrap pass each.
    selected = (
        [v.strip() for v in verifiers.split(",") if v.strip()]
        if verifiers
        else [cfg.risk.primary_verifier, "v2_text", "v4_image"]
    )

    results.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for policy in cfg.risk.sensitivity_harm_policies:
        table = rh1_discrimination_table(
            data,
            policy,
            verifiers=selected,
            n_bins=cfg.calibration.n_bins,
            n_bootstrap=cfg.stats.n_bootstrap,
            ci_level=cfg.stats.ci_level,
            seed=cfg.stats.bootstrap_seed,
        )
        if table.empty:
            console.print(f"[red]no matched candidates for policy {policy}[/red]")
            raise typer.Exit(1)
        suffix = "" if policy is cfg.risk.harm_policy else f"__{policy.value}"
        # The arms are matched under the PRIMARY policy only: `build_match` equalizes
        # counts by harm class, and the class membership moves with the policy. Under
        # non_improving the fit-harmful counts diverge by up to 502 rows between arms. The
        # sensitivity tables are therefore not matched contrasts and must say so in the
        # data, not only in prose a reader may not reach.
        table = table.assign(
            matched_under_policy=cfg.risk.harm_policy.value,
            arms_matched_for_this_policy=policy is cfg.risk.harm_policy,
        )
        path = results / f"rh1_discrimination{suffix}.csv"
        table.to_csv(path, index=False)
        written.append(path)
        if policy is cfg.risk.harm_policy:
            primary = table

    shown = primary[
        (primary["verifier_id"] == cfg.risk.primary_verifier) & (primary["metric"] == RH1_PRIMARY)
    ]
    rendered = Table(
        "engine",
        "donor",
        "zero-shot",
        "matched ref",
        "delta",
        "95% CI",
        "degraded",
        title=f"RH1 {RH1_PRIMARY} on {cfg.risk.primary_verifier}: zero-shot minus matched "
        "reference (negative = transfer ranks worse)",
    )
    for _, row in shown.sort_values(["held_out_engine", "donor_engine"]).iterrows():
        rendered.add_row(
            str(row["held_out_engine"]),
            str(row["donor_engine"]),
            f"{row['loeo_zero_shot']:.4f}",
            f"{row['matched_in_engine']:.4f}",
            f"{row['delta']:+.4f}",
            f"[{row['delta_ci_lower']:+.4f}, {row['delta_ci_upper']:+.4f}]",
            "[red]yes[/red]" if row["degraded_after_holm"] else "no",
        )
    console.print(rendered)
    for path in written:
        console.print(f"wrote {path}")
    console.print(f"  sources: {sources}")
    secondary = ", ".join(m for m in RH1_METRICS if m != RH1_PRIMARY)
    console.print(f"Secondary endpoints in the same tables: {secondary}")
    console.print(
        f"[yellow]Arms are matched under {cfg.risk.harm_policy.value} only. The other "
        "policies' tables carry arms_matched_for_this_policy=False.[/yellow]"
    )


@gate_app.command("h2-readiness")
def h2_readiness(
    generator: Annotated[
        str, typer.Option("--generator", help="The generator frozen by the selection rule")
    ] = "g3_edit_aware",
    study: Annotated[Path, typer.Option("--study")] = Path("results/generated/generators"),
    rh1_experiment: Annotated[
        str | None, typer.Option("--rh1", help="Experiment holding the match certificates")
    ] = "rh1_matched",
    alignment_sensitivity: Annotated[
        Path | None,
        typer.Option("--alignment-sensitivity", help="The EasyOCR support analysis"),
    ] = Path("results/generated/alignment_support_sensitivity.csv"),
    out: Annotated[Path, typer.Option("--out")] = Path("results/generated"),
) -> None:
    """Judge the frozen H2-readiness criteria on the held-out partition.

    Reads the ``--role evaluate`` study, not the development one. There is no partial
    pass: H2 is a single experiment that either can or cannot be run on this corpus.
    """
    import json

    from ocr_risk.experiments.readiness import (
        EngineCell,
        evaluate_h2_readiness,
        h2_readiness_criteria,
    )

    quality_path = study / "generator_quality__evaluate.csv"
    if not quality_path.is_file():
        raise typer.BadParameter(
            f"{quality_path} is missing; run `ocr-risk candidates study ... --role evaluate` "
            "with the frozen generator first"
        )
    quality = pd.read_csv(quality_path)
    cells = quality[(quality["generator_id"] == generator) & (quality["dataset_id"] == "ALL")]
    if cells.empty:
        scoped = sorted(
            quality[
                (quality["generator_id"] == generator)
                & quality["dataset_id"].astype(str).str.startswith("ALL:")
            ]["dataset_id"].unique()
        )
        if scoped:
            # The readiness floors were derived at 4 101-4 890 sites per engine over three
            # corpora. Judging a corpus-restricted generator against them would compare a
            # smaller, different population to a threshold that does not describe it.
            raise typer.BadParameter(
                f"generator {generator!r} is scoped to {scoped[0]} and cannot be judged "
                "against corpus-wide readiness thresholds; report it separately"
            )
        raise typer.BadParameter(f"no evaluate-role cells for generator {generator!r}")

    per_engine = [
        EngineCell(
            engine_id=str(row["engine_id"]),
            oracle_accepted=round(float(row["oracle_safe_coverage"]) * int(row["n_sites"])),
            oracle_safe_coverage=float(row["oracle_safe_coverage"]),
            n_nonidentity_candidates=int(row["n_nonidentity_candidates"]),
            n_sites=int(row["n_sites"]),
            n_clean_sites=int(row["n_clean_sites"]),
            clean_span_proposal_rate=float(row["clean_span_proposal_rate"]),
        )
        for _, row in cells.sort_values("engine_id").iterrows()
    ]

    store = ArtifactStore()
    all_matched: bool | None = None
    n_test_documents = 0
    if rh1_experiment:
        predict_run = store.latest(StageName.PREDICT, rh1_experiment)
        if predict_run is not None:
            payload = store.read_json(predict_run, "match_certificates.json")
            all_matched = bool(payload["all_matched"])
            n_test_documents = max(
                (int(c["zero_shot"]["n_evaluate_documents"]) for c in payload["certificates"]),
                default=0,
            )

    # Criterion E is measured, not asserted: the study's manifest names every generator
    # that was scored, and a challenge/hard-negative generator is not in the ladder. A
    # missing manifest makes the criterion unmeasurable rather than passed on faith --
    # the same unfireable-guard failure the report itself criticizes elsewhere.
    from ocr_risk.experiments.generator_ladder import GENERATOR_LADDER

    manifest_path = study / "generator_study__evaluate.json"
    all_natural: bool | None = None
    if manifest_path.is_file():
        scored = json.loads(manifest_path.read_text(encoding="utf-8")).get("generators", [])
        all_natural = all(str(g) in GENERATOR_LADDER for g in scored)
    criteria = h2_readiness_criteria(
        per_engine=per_engine,
        n_test_documents=n_test_documents,
        all_natural=all_natural,
        all_pairs_matched=all_matched,
        alignment_sensitivity_reported=bool(
            alignment_sensitivity and alignment_sensitivity.is_file()
        ),
    )
    outcome = evaluate_h2_readiness(
        criteria,
        notes=(
            "Thresholds are frozen in docs/h1_recovery/h2_readiness_protocol.md and derived "
            "from the Bentkus sample-size table, not from the development results.",
            f"Judged on {generator!r}, held-out documents only."
            + (
                " This is the frozen generator."
                if generator == "g3_edit_aware"
                else " SENSITIVITY: the frozen generator is g3_edit_aware."
            ),
        ),
        context={
            "generator": generator,
            "study": str(quality_path),
            "rh1_experiment": rh1_experiment or "",
            "n_test_documents": n_test_documents,
        },
    )
    # The frozen generator's verdict is the headline; any other generator's is a
    # sensitivity and is named so a reader cannot mistake one for the other.
    name = (
        "h2_readiness_gate.json"
        if generator == "g3_edit_aware"
        else f"h2_readiness_gate__{generator}.json"
    )
    path = outcome.save(out / name)

    rendered = Table("criterion", "threshold", "observed", "met", title="H2 readiness")
    for criterion in outcome.criteria:
        state = (
            "[yellow]not measurable[/yellow]"
            if not criterion.measurable
            else ("[green]yes[/green]" if criterion.met else "[red]no[/red]")
        )
        rendered.add_row(criterion.key, criterion.threshold, criterion.observed, state)
    console.print(rendered)
    colour = {"H2 READY": "green", "H2 NOT READY": "red", "H2 INCONCLUSIVE": "yellow"}
    console.print(f"[{colour[outcome.verdict]}]{outcome.verdict}[/{colour[outcome.verdict]}]")
    for note in outcome.notes:
        console.print(f"  [dim]{note}[/dim]")
    console.print(f"  wrote {path}")
    console.print(json.dumps({"verdict": outcome.verdict}))


@app.command("alignment-support")
def alignment_support(
    experiment: Annotated[
        str,
        typer.Option("--experiment", "-e", help="Whose align/sites/candidates chain to read"),
    ] = "pilot_loeo_zero_shot",
    policy: Annotated[str, typer.Option("--policy")] = "strict_worsening",
    out: Annotated[Path, typer.Option("--out")] = Path(
        "results/generated/alignment_support_sensitivity.csv"
    ),
) -> None:
    """The section-8 sensitivity table, from the artifact chain it names.

    Common support is the set of ground-truth tokens **every** engine resolved -- an
    engine-symmetric, ground-truth-free membership rule; the table then measures clean
    share, harmful rate and mean error distance on the full population and on that subset.
    This command exists because the table was first committed without the code that
    produced it: a tracked result needs a producer a reader can run.
    """
    from ocr_risk.analysis.alignment_support import support_sensitivity_table

    store = ArtifactStore()
    align_run = store.latest(StageName.ALIGN, experiment)
    sites_run = store.latest(StageName.SITES, experiment)
    candidates_run = store.latest(StageName.CANDIDATES, experiment)
    missing = [
        name
        for name, record in (
            ("align", align_run),
            ("sites", sites_run),
            ("candidates", candidates_run),
        )
        if record is None
    ]
    if missing:
        raise typer.BadParameter(f"experiment {experiment!r} is missing runs: {', '.join(missing)}")
    assert align_run and sites_run and candidates_run
    table = support_sensitivity_table(
        store.read_table(align_run, "alignments").to_pandas(),
        store.read_table(sites_run, "sites").to_pandas(),
        store.read_table(candidates_run, "candidates").to_pandas(),
        store.read_table(candidates_run, "labels").to_pandas(),
        HarmPolicy(policy),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    for engine, share in (
        table[table["population"] == "common_support"][["engine_id", "n_sites"]]
        .set_index("engine_id")["n_sites"]
        .items()
    ):
        full = table[(table["population"] == "full") & (table["engine_id"] == engine)][
            "n_sites"
        ].iloc[0]
        console.print(f"  {engine}: common support keeps {share}/{full} sites")
    console.print(f"wrote {out}")
    console.print(
        f"  sources: align={align_run.run_id} sites={sites_run.run_id} "
        f"candidates={candidates_run.run_id}"
    )


@app.command("alignment-twin")
def alignment_twin(
    experiment: Annotated[
        str,
        typer.Option("--experiment", "-e", help="Whose align/sites chain to read"),
    ] = "pilot_loeo_zero_shot",
    role: Annotated[
        str,
        typer.Option("--role", help="Which study pass's proposals View B audits"),
    ] = "evaluate",
    region_proposals: Annotated[
        Path | None,
        typer.Option("--region-proposals", help="CGV2 region proposals CSV (View B)"),
    ] = None,
    site_proposals: Annotated[
        Path | None,
        typer.Option("--site-proposals", help="CGV2 site proposals CSV (View B)"),
    ] = None,
    out_dir: Annotated[Path, typer.Option("--out-dir")] = Path("results/generated/cgv2"),
) -> None:
    """The CGV2 section-15 twin audit: View A on the alignment, View B on region projection.

    View A counts the R-37 twin signature per engine and site kind, emits the per-site
    signature table the sensitivity variants need, and censuses the adjacent-pair
    population by the same exact-slice rule the study uses. View B re-derives every
    region candidate from the canonical sites table and the linearized stream --
    recomputing, never trusting, the study's own gap and range bookkeeping. Mandatory
    before the confirmatory evaluation. The audit is role-scoped: each role's View B
    binds that role's proposals by hash, and the tables/gates commands read the audit
    from ``{out_dir}/audits/{role}``.
    """
    from ocr_risk.align.confidence import ConfidenceWeights
    from ocr_risk.analysis.alignment_twin import (
        component_audit,
        component_denominators,
        projection_audit,
        projection_summary,
        region_census,
        region_pair_audit,
        site_eligibility_audit,
        site_eligibility_denominators,
        site_projection_audit,
        twin_sites_table,
        twin_table,
    )
    from ocr_risk.canonical import rebuild_stream
    from ocr_risk.schemas.spans import CanonicalSpan

    if role not in {"calibrate", "evaluate"}:
        raise typer.BadParameter("--role must be 'calibrate' or 'evaluate'")
    role_suffix = "" if role == "calibrate" else f"__{role}"
    base_dir = out_dir
    out_dir = out_dir / "audits" / role
    site_proposals = site_proposals or base_dir / f"cgv2_proposals{role_suffix}.csv"
    region_proposals = region_proposals or base_dir / f"cgv2_region_proposals{role_suffix}.csv"

    store = ArtifactStore()
    align_run = store.latest(StageName.ALIGN, experiment)
    sites_run = store.latest(StageName.SITES, experiment)
    canonical_run = store.latest(StageName.CANONICALIZE, experiment)
    manifest_run = store.latest(StageName.MANIFEST, experiment)
    missing = [
        name
        for name, record in (
            ("align", align_run),
            ("sites", sites_run),
            ("canonicalize", canonical_run),
            ("manifest", manifest_run),
        )
        if record is None
    ]
    if missing:
        raise typer.BadParameter(f"experiment {experiment!r} is missing runs: {', '.join(missing)}")
    assert align_run and sites_run and canonical_run and manifest_run
    alignments = store.read_table(align_run, "alignments").to_pandas()
    sites = store.read_table(sites_run, "sites").to_pandas()
    gt_tokens = pd.DataFrame(
        [record.model_dump(mode="json") for record in store.read_records(manifest_run, "gt_tokens")]
    )
    resolved = _resolve_config(experiment, None)
    alignment_config = resolved.config.alignment

    streams: dict[tuple[str, str], str] = {}
    grouped: dict[tuple[str, str], list[CanonicalSpan]] = {}
    for span in store.read_records(canonical_run, "spans"):
        grouped.setdefault((span.document_id, span.engine_id), []).append(span)
    for key, spans in grouped.items():
        streams[key] = rebuild_stream(spans)

    out_dir.mkdir(parents=True, exist_ok=True)
    twins = twin_table(alignments, sites)
    twin_sites = twin_sites_table(alignments, sites)
    components = component_audit(
        alignments,
        gt_tokens,
        min_align_confidence=alignment_config.min_align_confidence,
        weights=ConfidenceWeights(
            char_agreement=alignment_config.weight_char_agreement,
            geometry=alignment_config.weight_geometry,
            uniqueness=alignment_config.weight_uniqueness,
        ),
    )
    component_counts = component_denominators(components)
    site_eligibility = site_eligibility_audit(sites, components, streams)
    site_counts = site_eligibility_denominators(site_eligibility)
    region_pairs = region_pair_audit(sites, streams, site_eligibility)
    regions = region_census(sites, streams, pair_audit=region_pairs)

    audit_frames: dict[str, pd.DataFrame] = {
        "alignment_twin.csv": twins,
        "alignment_twin_sites.csv": twin_sites,
        "alignment_component_audit.csv": components,
        "alignment_component_denominators.csv": component_counts,
        "alignment_site_audit.csv": site_eligibility,
        "alignment_site_denominators.csv": site_counts,
        "region_pair_audit.csv": region_pairs,
        "region_census.csv": regions,
    }

    site_audit = pd.DataFrame()
    if site_proposals.exists():
        proposals = pd.read_csv(site_proposals, keep_default_na=False)
        site_audit = site_projection_audit(proposals, sites, streams, site_eligibility)
        audit_frames["site_projection_audit.csv"] = site_audit
    else:
        console.print(f"[yellow]{site_proposals} not found; site-level View B skipped[/yellow]")

    region_audit = pd.DataFrame()
    region_summary = pd.DataFrame()
    if region_proposals.exists():
        proposals = pd.read_csv(region_proposals, keep_default_na=False)
        region_audit = projection_audit(
            proposals,
            sites,
            streams,
            site_eligibility,
            region_pairs,
        )
        region_summary = projection_summary(region_audit)
        audit_frames["region_projection_audit.csv"] = region_audit
        audit_frames["region_projection_summary.csv"] = region_summary
        for row in region_summary.itertuples(index=False):
            engine = str(row.engine_id)
            console.print(
                f"  {engine}: {row.projection_correct}/{row.n_region_candidates} "
                f"projections correct, {row.projection_mismatch} mismatched"
            )
    else:
        console.print(
            f"[yellow]{region_proposals} not found; View B skipped "
            "(run `ocr-risk cgv2 study` first)[/yellow]"
        )

    for row in twins.itertuples(index=False):
        if row.n_twin_signature:
            engine = str(row.engine_id)
            kind = str(row.site_kind)
            n_twin = int(str(row.n_twin_signature))
            n_sites = int(str(row.n_evaluable_sites))
            share = float(str(row.twin_share))
            console.print(f"  twin signature: {engine}/{kind} {n_twin}/{n_sites} ({share:.1%})")

    for name, frame in audit_frames.items():
        frame.to_csv(out_dir / name, index=False)

    component_accounted = bool(
        len(components) == int(component_counts["n_components_before_exclusion"].sum())
        if not component_counts.empty
        else alignments.empty
    )
    site_accounted = bool(
        len(site_eligibility) == int(site_counts["n_sites_before_exclusion"].sum())
        if not site_counts.empty
        else sites.empty
    )
    region_accounted = bool(
        (
            regions["n_pairs_before_exclusion"]
            == regions[
                [
                    "n_pairs_eligible_after_exclusion",
                    "n_pairs_ambiguous",
                    "n_pairs_excluded",
                    "n_pairs_unresolved",
                ]
            ].sum(axis=1)
        ).all()
    )
    site_projection_accounted = site_proposals.exists() and (
        site_audit.empty
        or bool(site_audit["state"].isin({"eligible", "ambiguous", "excluded", "unresolved"}).all())
    )
    region_projection_accounted = region_proposals.exists() and (
        region_audit.empty
        or bool(
            region_audit["state"].isin({"eligible", "ambiguous", "excluded", "unresolved"}).all()
        )
    )
    eligible_site_projection_pass = site_audit.empty or bool(
        site_audit[site_audit["state"] == "eligible"][["projection_correct", "label_valid"]]
        .astype(bool)
        .all(axis=None)
    )
    eligible_region_projection_pass = region_audit.empty or bool(
        region_audit[region_audit["state"] == "eligible"][["projection_correct", "label_valid"]]
        .astype(bool)
        .all(axis=None)
    )
    checks: dict[str, bool] = {
        "component_denominators_reconcile": component_accounted,
        "site_denominators_reconcile": site_accounted,
        "region_denominators_reconcile": region_accounted,
        "site_projection_rows_accounted": site_projection_accounted,
        "region_projection_rows_accounted": region_projection_accounted,
        "eligible_site_projections_valid": eligible_site_projection_pass,
        "eligible_region_projections_valid": eligible_region_projection_pass,
    }
    audit_gate: dict[str, object] = {
        "schema_version": "cgv2-alignment-audit-v2",
        "experiment": experiment,
        "role": role,
        "input_runs": {
            "align": align_run.run_id,
            "sites": sites_run.run_id,
            "canonicalize": canonical_run.run_id,
            "manifest": manifest_run.run_id,
        },
        "inputs": {
            "site_proposals": str(site_proposals),
            "site_proposals_sha256": file_sha256(site_proposals)
            if site_proposals.exists()
            else None,
            "region_proposals": str(region_proposals),
            "region_proposals_sha256": file_sha256(region_proposals)
            if region_proposals.exists()
            else None,
            # Bound to the audited tables themselves, not only to the proposals: a
            # re-run align/sites chain can leave proposal bytes identical while the
            # component states the audit certified go stale.
            "alignments_sha256": file_sha256(
                store.path_of(align_run, store.output_ref(align_run, "alignments.parquet"))
            ),
            "sites_sha256": file_sha256(
                store.path_of(sites_run, store.output_ref(sites_run, "sites.parquet"))
            ),
            "spans_sha256": file_sha256(
                store.path_of(canonical_run, store.output_ref(canonical_run, "spans.parquet"))
            ),
            "gt_tokens_sha256": file_sha256(
                store.path_of(manifest_run, store.output_ref(manifest_run, "gt_tokens.parquet"))
            ),
        },
        "counts": {
            "components": len(components),
            "site_candidates": len(site_audit),
            "region_candidates": len(region_audit),
            "region_pairs_before_exclusion": int(regions["n_pairs_before_exclusion"].sum()),
            "region_pairs_eligible_after_exclusion": int(
                regions["n_pairs_eligible_after_exclusion"].sum()
            ),
            "sites_before_exclusion": len(site_eligibility),
            "sites_eligible_after_exclusion": int((site_eligibility["state"] == "eligible").sum()),
            "sites_ambiguous": int((site_eligibility["state"] == "ambiguous").sum()),
            "sites_excluded": int((site_eligibility["state"] == "excluded").sum()),
            "sites_unresolved": int((site_eligibility["state"] == "unresolved").sum()),
            "newly_ambiguous_full_uniqueness_components": int(
                components["newly_ambiguous_full_uniqueness"].astype(bool).sum()
            ),
            "site_candidates_eligible": int((site_audit["state"] == "eligible").sum())
            if not site_audit.empty
            else 0,
            "region_candidates_eligible": int((region_audit["state"] == "eligible").sum())
            if not region_audit.empty
            else 0,
        },
        "checks": checks,
    }
    audit_gate["outputs"] = {
        name: {"rows": len(frame), "sha256": file_sha256(out_dir / name)}
        for name, frame in audit_frames.items()
    }
    audit_gate["overall_pass"] = all(checks.values())
    (out_dir / "alignment_audit_gate.json").write_text(
        json.dumps(audit_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    console.print(
        f"wrote {out_dir}/alignment_twin.csv, alignment_component_audit.csv, "
        "site_projection_audit.csv, region_projection_audit.csv, and denominators"
    )


@app.command("recovery")
def recovery(
    study: Annotated[Path, typer.Option("--study")] = Path("results/generated/generators"),
    rh1_table: Annotated[Path, typer.Option("--rh1-table")] = Path(
        "results/generated/rh1_discrimination.csv"
    ),
    rh1_experiment: Annotated[str, typer.Option("--rh1")] = "rh1_matched",
    pool: Annotated[str, typer.Option("--pool")] = "pilot_loeo_zero_shot",
    metric: Annotated[str, typer.Option("--metric")] = "roc_auc",
    verifier: Annotated[str, typer.Option("--verifier")] = "v6_full",
    out: Annotated[Path, typer.Option("--out")] = Path("results/generated"),
) -> None:
    """Assemble the recovery-phase figures and tables from artifacts already produced.

    Recovery-phase evidence, not manuscript figures. Every ground-truth oracle quantity is
    labelled as one on the figure itself.
    """
    from ocr_risk.analysis.recovery_figures import (
        plot_candidate_quality,
        plot_matched_discrimination,
        plot_oracle_potential,
        plot_score_distributions,
    )
    from ocr_risk.experiments.readiness import MIN_ORACLE_COVERAGE

    out.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    def spec(name: str, title: str, caption: str, runs: tuple[str, ...]) -> FigureSpec:
        return FigureSpec(
            path=out / name,
            title=title,
            caption=caption,
            synthetic=False,
            source_runs=runs,
            config_sha256="",
        )

    for role, suffix in (("evaluate", "__evaluate"), ("calibrate", "")):
        quality_path = study / f"generator_quality{suffix}.csv"
        if not quality_path.is_file():
            continue
        quality = pd.read_csv(quality_path)
        # A generator scoped to a subset of corpora is measured on a different population
        # and is not plotted beside the corpus-wide rungs. Saying so in the caption is the
        # difference between "excluded for a reason" and "silently missing".
        scoped = sorted(
            quality[quality["dataset_id"].astype(str).str.startswith("ALL:")][
                ["generator_id", "dataset_id"]
            ]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        excluded = (
            "  Not plotted: "
            + ", ".join(f"{g} (scored on {d.split(':', 1)[1]} only)" for g, d in scoped)
            + " — a different population from the corpus-wide rungs."
            if scoped
            else ""
        )
        produced.append(
            plot_candidate_quality(
                quality,
                spec(
                    f"R1_candidate_quality_{role}.png",
                    f"R1 — candidate quality by generator ({role} documents)",
                    "Shares of the non-identity pool. n is the pool a share is taken of: a "
                    "generator that proposes 80x less can hold the same beneficial share."
                    + excluded,
                    (str(quality_path),),
                ),
            )
        )
        produced.append(
            plot_oracle_potential(
                quality,
                spec(
                    f"R2_oracle_potential_{role}.png",
                    f"R2 — what a perfect verifier could reach ({role} documents)",
                    "Upper bounds a deployed system cannot attain. The dashed line is the "
                    "pre-registered H2-readiness floor, frozen before this partition was "
                    "scored." + excluded,
                    (str(quality_path),),
                ),
                threshold=MIN_ORACLE_COVERAGE,
            )
        )

    if rh1_table.is_file():
        table = pd.read_csv(rh1_table)
        produced.append(
            plot_matched_discrimination(
                table,
                spec(
                    f"R3_matched_discrimination_{metric}.png",
                    f"R3 — matched in-engine reference vs LOEO zero-shot ({metric})",
                    f"Verifier {verifier}. Diamonds are the pooled per-engine contrast "
                    "under a joint document resample, small points the three individual "
                    "substitutions, red survived Holm across the four target engines.",
                    (str(rh1_table),),
                ),
                metric=metric,
                verifier=verifier,
            )
        )

        store = ArtifactStore()
        try:
            data, sources = _load(store, rh1_experiment, pool_experiment=pool)
        except FileNotFoundError:
            data = None
        if data is not None:
            from ocr_risk.analysis.tables import _harmful
            from ocr_risk.schemas.enums import HarmPolicy

            frame = data.predictions.merge(
                data.labels[["candidate_id", "outcome_if_accepted"]],
                on="candidate_id",
                how="inner",
            ).merge(
                data.candidates[["candidate_id", "is_synthetic_hard_negative"]],
                on="candidate_id",
                how="left",
            )
            frame = data.natural(frame)
            frame = frame[frame["verifier_id"] == verifier]
            harmful = _harmful(frame, HarmPolicy.STRICT_WORSENING)
            frame = frame.assign(
                arm=frame["fold_id"].str.split(":").str[0],
                held_out_engine=frame["fold_id"]
                .str.rsplit(":", n=1)
                .str[-1]
                .str.split("<-")
                .str[0],
                score=frame["calibrated_score"].fillna(frame["raw_score"]).astype(float),
                **{"class": ["harmful" if h else "safe" for h in harmful]},
            )
            produced.append(
                plot_score_distributions(
                    frame,
                    spec(
                        "R4_score_distributions.png",
                        f"R4 — score distributions by class and arm ({verifier})",
                        "The picture behind every ranking statistic. An arm whose scores "
                        "collapsed onto one value shows it here and in no summary number. "
                        f"Sources: {sources}",
                        (rh1_experiment, pool),
                    ),
                    arms=("loeo_zero_shot", "matched_in_engine"),
                )
            )

    for path in produced:
        console.print(f"wrote {path}")
    if not produced:
        console.print("[yellow]no inputs found; nothing to plot[/yellow]")


@gate_app.command("generator")
def generator_gate(
    baseline: Annotated[str, typer.Option("--baseline")] = "g0_lexical",
    best: Annotated[str, typer.Option("--best")] = "g3_edit_aware",
    study: Annotated[Path, typer.Option("--study")] = Path("results/generated/generators"),
    out: Annotated[Path, typer.Option("--out")] = Path("results/generated"),
) -> None:
    """Did the generator ladder move the candidate pool, relative to the pilot's generator?

    A narrower question than H2 readiness, and its criteria are relative rather than
    absolute. Unlike the H2 criteria they were written after the measurements; the H2
    verdict does not rest on them.
    """
    import json

    from ocr_risk.experiments.readiness import (
        EngineCell,
        evaluate_generator_gate,
        generator_criteria,
        h2_readiness_criteria,
    )

    quality = pd.read_csv(study / "generator_quality__evaluate.csv")
    pooled = quality[quality["dataset_id"] == "ALL"]

    def cells(generator_id: str) -> list[EngineCell]:
        rows = pooled[pooled["generator_id"] == generator_id].sort_values("engine_id")
        if rows.empty:
            raise typer.BadParameter(
                f"no corpus-wide evaluate-role cells for {generator_id!r}; a generator "
                "scoped to a subset of corpora is reported separately, not gated"
            )
        return [
            EngineCell(
                engine_id=str(row["engine_id"]),
                oracle_accepted=round(float(row["oracle_safe_coverage"]) * int(row["n_sites"])),
                oracle_safe_coverage=float(row["oracle_safe_coverage"]),
                n_nonidentity_candidates=int(row["n_nonidentity_candidates"]),
                n_sites=int(row["n_sites"]),
                n_clean_sites=int(row["n_clean_sites"]),
                clean_span_proposal_rate=float(row["clean_span_proposal_rate"]),
            )
            for _, row in rows.iterrows()
        ]

    def mean_of(generator_id: str, column: str) -> float:
        return float(pooled[pooled["generator_id"] == generator_id][column].mean())

    frozen = {
        c.key: c
        for c in h2_readiness_criteria(
            per_engine=cells(best),
            n_test_documents=999,
            all_natural=True,
            all_pairs_matched=True,
            alignment_sensitivity_reported=True,
        )
    }
    criteria = generator_criteria(
        baseline=cells(baseline),
        best=cells(best),
        baseline_id=baseline,
        best_id=best,
        baseline_clean_span_rate=mean_of(baseline, "clean_span_proposal_rate"),
        best_clean_span_rate=mean_of(best, "clean_span_proposal_rate"),
        baseline_repair_opportunity=mean_of(baseline, "error_repair_opportunity"),
        best_repair_opportunity=mean_of(best, "error_repair_opportunity"),
        meets_h2_repair=frozen["A_repair_opportunity"].met,
        meets_h2_coverage=frozen["B_oracle_safe_coverage"].met,
    )
    outcome = evaluate_generator_gate(
        criteria,
        notes=(
            f"{best} against {baseline}, held-out documents, mean over engines where a "
            "mean is used.",
            "These criteria are relative and were written after the measurements. The "
            "frozen absolute criteria are the H2-readiness ones.",
        ),
    )
    path = outcome.save(out / "generator_gate.json")

    rendered = Table("criterion", "threshold", "observed", "met", title="generator gate")
    for criterion in outcome.criteria:
        rendered.add_row(
            criterion.key,
            criterion.threshold,
            criterion.observed,
            "[green]yes[/green]" if criterion.met else "[red]no[/red]",
        )
    console.print(rendered)
    console.print(f"[bold]{outcome.verdict}[/bold]")
    for note in outcome.notes:
        console.print(f"  [dim]{note}[/dim]")
    console.print(f"  wrote {path}")
    console.print(json.dumps({"verdict": outcome.verdict}))


@app.command("recovery-tables")
def recovery_tables(
    study: Annotated[Path, typer.Option("--study")] = Path("results/generated/generators"),
    rh1_table: Annotated[Path, typer.Option("--rh1-table")] = Path(
        "results/generated/rh1_discrimination.csv"
    ),
    gates: Annotated[Path, typer.Option("--gates")] = Path("results/generated"),
    metric: Annotated[str, typer.Option("--metric")] = "roc_auc",
    verifier: Annotated[str, typer.Option("--verifier")] = "v6_full",
    out: Annotated[Path, typer.Option("--out")] = Path("results/generated"),
) -> None:
    """Assemble the three recovery-phase tables from artifacts already produced.

    R1 candidate-generator statistics, R2 H2-readiness statistics, R3 the matched-handicap
    transfer comparison. Each is a projection of a larger artifact, kept narrow enough to
    read; the artifacts they came from are named in a sidecar so no number is orphaned.
    """
    import json

    out.mkdir(parents=True, exist_ok=True)
    sources: dict[str, str] = {}
    written: list[Path] = []

    evaluate_path = study / "generator_quality__evaluate.csv"
    if evaluate_path.is_file():
        quality = pd.read_csv(evaluate_path)
        pooled = quality[quality["dataset_id"].astype(str).str.startswith("ALL")].copy()
        pooled["oracle_accepted"] = (
            (pooled["oracle_safe_coverage"] * pooled["n_sites"]).round().astype(int)
        )
        columns = [
            "generator_id",
            "engine_id",
            "n_sites",
            "n_clean_sites",
            "n_error_sites",
            "n_nonidentity_candidates",
            "beneficial_rate",
            "harmful_rate",
            "no_change_rate",
            "error_repair_opportunity",
            "clean_span_proposal_rate",
            "exact_correction_rate",
            "partial_improvement_rate",
            "oracle_safe_coverage",
            "oracle_repair_recall",
            "oracle_accepted",
        ]
        path = out / "R1_generator_statistics.csv"
        pooled[columns].sort_values(["generator_id", "engine_id"]).to_csv(path, index=False)
        written.append(path)
        sources["R1"] = str(evaluate_path)

        # R2: the readiness arithmetic, one row per engine, with the frozen thresholds
        # beside the measurement so a reader never has to look them up elsewhere.
        from ocr_risk.experiments.readiness import MIN_ORACLE_ACCEPTED, MIN_ORACLE_COVERAGE
        from ocr_risk.metrics.precision import minimum_accepted_edits

        rows: list[dict[str, object]] = []
        for _, row in pooled.sort_values(["generator_id", "engine_id"]).iterrows():
            available = int(row["oracle_accepted"])
            for epsilon in (0.10, 0.05, 0.02, 0.01, 0.005, 0.001):
                required = minimum_accepted_edits(epsilon, 0.1, 0.0)
                at_half = minimum_accepted_edits(epsilon, 0.1, epsilon / 2.0)
                # The controller's convention: delta spent per threshold test, not flat.
                # Both are recorded because the frozen protocol's thresholds were derived
                # at flat delta and stay the criterion of record, while the controller
                # certifies at the split one (amendment A10.4).
                grid_required = minimum_accepted_edits(epsilon, 0.1, 0.0, n_thresholds=200)
                grid_at_half = minimum_accepted_edits(epsilon, 0.1, epsilon / 2.0, n_thresholds=200)
                rows.append(
                    {
                        "generator_id": row["generator_id"],
                        "engine_id": row["engine_id"],
                        "epsilon": epsilon,
                        "n_accepted_available_oracle": available,
                        "n_accepted_required_if_clean": required,
                        "n_accepted_required_at_half_epsilon": at_half,
                        # The clean column is the best case, not the planning case: a
                        # method that accepts nothing harmful is not the case you size a
                        # study for. Quoting `resolvable` alone would overstate.
                        "resolvable": bool(required is not None and available >= required),
                        "resolvable_at_half_epsilon": bool(
                            at_half is not None and available >= at_half
                        ),
                        "n_required_if_clean__grid_delta": grid_required,
                        "n_required_at_half_epsilon__grid_delta": grid_at_half,
                        "oracle_safe_coverage": float(row["oracle_safe_coverage"]),
                        "threshold_oracle_accepted": MIN_ORACLE_ACCEPTED,
                        "threshold_oracle_coverage": MIN_ORACLE_COVERAGE,
                    }
                )
        path = out / "R2_h2_readiness_statistics.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        written.append(path)
        sources["R2"] = str(evaluate_path)

    if rh1_table.is_file():
        table = pd.read_csv(rh1_table)
        shown = table[table["verifier_id"] == verifier]
        columns = [
            "held_out_engine",
            "donor_engine",
            "metric",
            "loeo_zero_shot",
            "matched_in_engine",
            "delta",
            "delta_ci_lower",
            "delta_ci_upper",
            "p_value",
            "holm_adjusted_p",
            "degraded_after_holm",
            "minimum_detectable_effect",
            "n_candidates",
            "n_documents",
            "base_rate_safe",
        ]
        path = out / "R3_matched_transfer.csv"
        shown[columns].sort_values(["metric", "held_out_engine", "donor_engine"]).to_csv(
            path, index=False
        )
        written.append(path)
        sources["R3"] = str(rh1_table)

    for name in ("h2_readiness_gate.json", "generator_gate.json", "rh1_gate.json"):
        if (gates / name).is_file():
            sources[name] = str(gates / name)

    sidecar = out / "recovery_tables_sources.json"
    sidecar.write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in written:
        console.print(f"wrote {path} ({len(pd.read_csv(path))} rows)")
    console.print(f"wrote {sidecar}")


@gate_app.command("rh1")
def rh1_gate(
    table_path: Annotated[Path, typer.Option("--table")] = Path(
        "results/generated/rh1_discrimination.csv"
    ),
    verifier: Annotated[str, typer.Option("--verifier")] = "v6_full",
    out: Annotated[Path, typer.Option("--out")] = Path("results/generated"),
) -> None:
    """Judge RH1 on the pre-registered primary endpoint, per target engine.

    Read from the **pooled** row, which marginalizes the donor substitution out: the three
    donors are three views of one comparison, and counting an engine on the strength of its
    most favourable donor would reintroduce exactly the arbitrary choice the averaging
    removes. The per-donor rows are diagnostics and carry no adjusted p.

    The harm-policy sensitivity tables are read alongside: the protocol pre-commits that if
    the policies disagree, the verdict is PARTIALLY SUPPORTED and the disagreement is the
    headline.
    """
    import json

    from ocr_risk.analysis.rh1_discrimination import RH1_METRICS, RH1_PRIMARY
    from ocr_risk.experiments.readiness import Criterion, evaluate_rh1_gate

    if not table_path.is_file():
        raise typer.BadParameter(f"{table_path} is missing; run `ocr-risk analyze rh1` first")
    table = pd.read_csv(table_path)
    full = table[(table["verifier_id"] == verifier) & (table["metric"] == RH1_PRIMARY)]
    primary = full[full["aggregation"] == "pooled"]
    if primary.empty:
        raise typer.BadParameter(
            f"no pooled {RH1_PRIMARY} rows for verifier {verifier!r}; the verdict is read "
            "from the donor-marginalized row, not from a per-donor one"
        )

    engines = sorted(full["held_out_engine"].unique())
    # An engine is measurable only if it has exactly one pooled row with a finite,
    # non-degenerate interval. A missing or degenerate row is evidence that was not
    # produced, which the protocol's own section 5 routes to INCONCLUSIVE rather than
    # silently counting as not-degraded.
    degraded: list[str] = []
    unmeasurable: list[str] = []
    for engine in engines:
        row = primary[primary["held_out_engine"] == engine]
        upper = float(row.iloc[0]["delta_ci_upper"]) if len(row) == 1 else float("nan")
        if len(row) != 1 or bool(row.iloc[0]["degenerate_interval"]) or not math.isfinite(upper):
            unmeasurable.append(engine)
        elif bool(row.iloc[0]["degraded_after_holm"]):
            degraded.append(engine)
    per_donor = full[full["aggregation"] == "per_donor"]
    any_donor = [
        engine
        for engine in engines
        if bool((per_donor[per_donor["held_out_engine"] == engine]["delta_ci_upper"] < 0).any())
    ]

    supporting: list[Criterion] = []
    for metric in RH1_METRICS:
        if metric == RH1_PRIMARY:
            continue
        # Pooled only: the per-donor rows carry degraded_after_holm=False by construction,
        # so including them in the `.all()` would force every secondary count to zero.
        rows = table[
            (table["verifier_id"] == verifier)
            & (table["metric"] == metric)
            & (table["aggregation"] == "pooled")
        ]
        count = sum(
            bool(rows[rows["held_out_engine"] == e]["degraded_after_holm"].all()) for e in engines
        )
        supporting.append(
            Criterion(
                key=f"secondary_{metric}",
                question=f"Does {metric} degrade under transfer, per engine, after Holm?",
                threshold="reported, not decisive",
                observed=f"{count} of {len(engines)} engines",
                met=count > 0,
                measurable=not rows.empty,
            )
        )

    # A sensitivity table with no pooled primary rows is evidence that was not produced,
    # not a measured zero. Reading it as zero manufactured a disagreement during the
    # window in which the primary table is fresh and a sensitivity table is not.
    policies: dict[str, int] = {}
    unreadable: list[str] = []
    sensitivity_paths = sorted(table_path.parent.glob("rh1_discrimination__*.csv"))
    for path in sensitivity_paths:
        name = path.stem.split("__", 1)[1]
        other = pd.read_csv(path)
        rows = other[
            (other["verifier_id"] == verifier)
            & (other["metric"] == RH1_PRIMARY)
            & (other["aggregation"] == "pooled")
        ]
        if rows.empty:
            unreadable.append(name)
            continue
        policies[name] = sum(
            bool(rows[rows["held_out_engine"] == e]["degraded_after_holm"].all())
            for e in sorted(rows["held_out_engine"].unique())
        )
    sensitivity_measurable = bool(sensitivity_paths) and not unreadable
    disagree = sensitivity_measurable and len({len(degraded), *policies.values()}) > 1
    agreement_observed = f"primary={len(degraded)}; " + ", ".join(
        f"{name}={count}" for name, count in sorted(policies.items())
    )
    if not sensitivity_paths:
        agreement_observed += "; no sensitivity tables found"
    if unreadable:
        agreement_observed += "; no pooled rows in: " + ", ".join(unreadable)
    supporting.append(
        Criterion(
            key="harm_policy_agreement",
            question="Do the harm policies agree on how many engines degraded?",
            threshold="pre-committed: disagreement makes the verdict PARTIALLY SUPPORTED",
            observed=agreement_observed,
            met=not disagree,
            measurable=sensitivity_measurable,
        )
    )

    n_donors = int(per_donor.groupby("held_out_engine").size().max()) if not per_donor.empty else 0
    notes = [
        f"Read from the pooled row, which marginalizes over {n_donors} donor substitutions "
        "per target engine under a joint document resample.",
        f"Engines whose interval excludes zero on at least one individual donor: "
        f"{any_donor or 'none'} (diagnostic; unadjusted).",
        f"Primary endpoint {RH1_PRIMARY} on {verifier}, frozen in "
        "docs/h1_recovery/h2_readiness_protocol.md section 4.",
        "Sensitivity tables are reported as sensitivity, not as matched contrasts "
        "(amendment A4: the arms are matched under the primary harm policy only).",
    ]
    if disagree:
        # The pre-committed rule, applied by evaluate_rh1_gate itself: the disagreement is
        # the headline, whatever the engine counts say. The H1 pilot's verdict flipped
        # across harm policies with no rule for what to conclude.
        notes.insert(
            0,
            "The harm policies disagree on how many engines degraded; per the rule frozen "
            "before the run, the verdict is PARTIALLY SUPPORTED and this disagreement is "
            "the headline.",
        )
    if unmeasurable:
        notes.insert(0, f"Engines with missing or degenerate pooled intervals: {unmeasurable}.")
    outcome = evaluate_rh1_gate(
        len(degraded),
        len(engines),
        primary_metric=RH1_PRIMARY,
        supporting=supporting,
        notes=notes,
        context={
            "verifier": verifier,
            "engines_degraded": degraded,
            "engines_unmeasurable": unmeasurable,
            "harm_policy_disagreement": bool(disagree),
            "table": str(table_path),
        },
        harm_policy_disagreement=disagree,
        n_engines_unmeasurable=len(unmeasurable),
        sensitivity_evidence_missing=not sensitivity_measurable,
    )
    path = outcome.save(out / "rh1_gate.json")

    rendered = Table("criterion", "threshold", "observed", "met", title="RH1 gate")
    for criterion in outcome.criteria:
        state = (
            "[yellow]n/a[/yellow]"
            if not criterion.measurable
            else ("[green]yes[/green]" if criterion.met else "[red]no[/red]")
        )
        rendered.add_row(criterion.key, criterion.threshold, criterion.observed, state)
    console.print(rendered)
    console.print(f"[bold]{outcome.verdict}[/bold]")
    for note in outcome.notes:
        console.print(f"  [dim]{note}[/dim]")
    console.print(f"  wrote {path}")
    console.print(json.dumps({"verdict": outcome.verdict}))
