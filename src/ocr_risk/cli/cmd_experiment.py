"""``ocr-risk experiment`` — run a full experiment, and inspect what it produced."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ocr_risk.config import load_config
from ocr_risk.experiments import run_experiment
from ocr_risk.io.artifacts import ArtifactStore
from ocr_risk.schemas.enums import StageName

app = typer.Typer(name="experiment", help="Run and inspect experiments.", no_args_is_help=True)
console = Console()

_PIPELINE_STAGES = (
    StageName.MANIFEST,
    StageName.CANONICALIZE,
    StageName.ALIGN,
    StageName.SITES,
    StageName.CANDIDATES,
    StageName.PREDICT,
    StageName.DECIDE,
)


@app.command("run")
def run(
    config_path: Annotated[Path, typer.Option("--config", "-c", help="Experiment config")],
    overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
    no_crops: Annotated[
        bool, typer.Option("--no-crops", help="Skip materializing image crops")
    ] = False,
) -> None:
    """Run every pipeline stage, writing one immutable artifact per stage."""
    resolved = load_config(config_path, overrides)
    cfg = resolved.config
    console.print(f"[bold]{cfg.name}[/bold]  config_sha256={resolved.sha256[:12]}")
    if cfg.synthetic:
        console.print(
            "[yellow]SYNTHETIC — outputs are architecture validation, not results[/yellow]"
        )

    started = time.monotonic()
    log: list[str] = []
    state = run_experiment(resolved, materialize_crops=not no_crops, progress=log)
    elapsed = time.monotonic() - started

    for line in log:
        console.print(f"  {line}")
    console.print(
        f"[green]complete[/green] in {elapsed:.1f}s  "
        f"{len(state.fold_results)} fold x method results"
    )
    _print_thresholds(state)


@app.command("status")
def status(
    experiment: Annotated[str, typer.Option("--experiment", "-e", help="Experiment name")],
) -> None:
    """Show which pipeline stages have artifacts for an experiment."""
    store = ArtifactStore()
    table = Table("stage", "status", "run id", "outputs", "rows")
    complete = True
    for stage in _PIPELINE_STAGES:
        record = store.latest(stage, experiment)
        if record is None:
            table.add_row(stage.value, "[red]MISSING[/red]", "-", "-", "-")
            complete = False
            continue
        rows = sum(ref.n_rows or 0 for ref in record.outputs)
        intact = store.verify_outputs(record)
        table.add_row(
            stage.value,
            "[green]OK[/green]" if intact else "[red]CORRUPTED[/red]",
            record.run_id,
            str(len(record.outputs)),
            str(rows) if rows else "-",
        )
        complete = complete and intact
    console.print(table)
    if not complete:
        raise typer.Exit(code=1)


def _print_thresholds(state: object) -> None:
    """Show the selected operating points, including where the controller abstained."""
    results = getattr(state, "fold_results", [])
    rows = [t for r in results for t in r.thresholds]
    if not rows:
        return

    table = Table(
        "fold",
        "method",
        "epsilon",
        "tau",
        "cal coverage",
        "cal risk",
        "risk bound",
        "feasible",
        title="selected thresholds (fitted on the calibration split only)",
    )
    for row in rows[:24]:
        table.add_row(
            str(row["fold_id"]).replace("loeo_zero_shot:", ""),
            str(row["method_id"]).split("|")[0],
            f"{float(row['epsilon']):.3f}",
            f"{float(row['tau']):.4f}",
            f"{float(row['coverage']):.3f}",
            f"{float(row['observed_risk']):.4f}",
            f"{float(row['risk_upper_bound']):.4f}",
            "yes" if row["feasible"] else "[yellow]abstain[/yellow]",
        )
    console.print(table)
    if len(rows) > 24:
        console.print(f"  ... and {len(rows) - 24} more (see the decide-stage artifact)")


@app.command("rh1")
def rh1(
    config_path: Annotated[Path, typer.Option("--config", "-c", help="RH1 experiment config")],
    source: Annotated[
        str,
        typer.Option("--source", help="Experiment whose candidates artifact supplies the pool"),
    ] = "pilot_loeo_zero_shot",
    overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    """Run the matched-handicap transfer study on an existing candidate pool.

    Reuses ``--source``'s candidates artifact rather than regenerating one, so the only
    thing that differs from the H1 pilot is the experimental design. Regenerating the pool
    at the same time would leave any difference unattributable between the two changes.
    """
    import json

    from ocr_risk.experiments.rh1 import load_pool, run_rh1
    from ocr_risk.io.hashing import canonical_hash
    from ocr_risk.provenance import capture_git_state

    resolved = load_config(config_path, overrides)
    cfg = resolved.config
    store = ArtifactStore()

    pool = load_pool(store, source, expected_partition_id=cfg.splits.partition_id)
    candidates_run = store.latest(StageName.CANDIDATES, source)
    assert candidates_run is not None
    frame, partition = pool.frame, pool.partition
    console.print(
        f"[bold]{cfg.name}[/bold]  pool from {source}: {len(frame)} candidates, "
        f"{len(pool.crops)} crops"
    )

    started = time.monotonic()
    seen: set[str] = set()

    def progress(line: str) -> None:
        pair = line.split(" ")[0]
        if pair not in seen:
            seen.add(pair)
            console.print(f"  [dim]pair {pair}[/dim]")

    run = run_rh1(
        cfg,
        partition,
        frame,
        pool.bundles,
        pool.conf_samples,
        crops=pool.crops,
        progress=progress,
    )
    elapsed = time.monotonic() - started

    predictions = [p for r in run.fold_results for p in r.predictions]
    decisions = [d for r in run.fold_results for d in r.decisions]
    with store.begin(
        stage=StageName.PREDICT,
        config_sha256=resolved.sha256,
        resolved_config=resolved.mapping,
        experiment=cfg.name,
        git_commit=capture_git_state().commit,
    ) as writer:
        writer.inherit_from(candidates_run)
        writer.write_records("predictions", predictions)
        writer.write_json(
            "match_certificates.json",
            {
                "all_matched": run.all_matched,
                "n_pairs": len(run.certificates),
                "source_experiment": source,
                "source_candidates_run": candidates_run.run_id,
                "certificates": [c.as_dict() for c in run.certificates],
            },
        )
        writer.write_json(
            "folds.json",
            [
                {
                    "fold_id": r.fold_id,
                    "method_id": r.method_id,
                    "calibrator_id": r.calibrator_id,
                    "diagnostics": r.diagnostics,
                    "split": r.descriptor.model_dump(mode="json"),
                }
                for r in run.fold_results
            ],
        )
        writer.set_folds([r.descriptor for r in run.fold_results])
        writer.mark_synthetic(cfg.synthetic)
    predict_run = writer.record
    assert predict_run is not None

    with store.begin(
        stage=StageName.DECIDE,
        config_sha256=resolved.sha256,
        resolved_config=resolved.mapping,
        experiment=cfg.name,
        git_commit=capture_git_state().commit,
    ) as writer:
        writer.inherit_from(predict_run)
        writer.write_records("decisions", decisions)
        writer.write_json("thresholds.json", [t for r in run.fold_results for t in r.thresholds])
        writer.set_folds([r.descriptor for r in run.fold_results])
        writer.mark_synthetic(cfg.synthetic)

    table = Table(
        "pair",
        "matched",
        "fit n",
        "fit harmful",
        "cal n",
        "eval n",
        title="match certificates",
    )
    for certificate in run.certificates:
        left = certificate.zero_shot
        table.add_row(
            certificate.pair_id,
            "[green]yes[/green]" if certificate.matched else "[red]NO[/red]",
            str(left.n_fit_candidates),
            str(left.n_fit_harmful),
            str(left.n_calibrate_candidates),
            str(left.n_evaluate_candidates),
        )
    console.print(table)
    console.print(
        f"[green]complete[/green] in {elapsed:.1f}s  {len(run.fold_results)} arm x method "
        f"results over {len(run.pairs)} pairs  (digest {canonical_hash(sorted(seen))[:12]})"
    )
    console.print(json.dumps({"all_matched": run.all_matched}))
