"""``ocr-risk provenance`` — trace an artifact back to its source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from typer.core import TyperGroup

from ocr_risk.io.artifacts import ArtifactStore
from ocr_risk.provenance import build_lineage, render_lineage


class _TraceByDefault(TyperGroup):
    """Route a bare argument to ``trace`` without shadowing the real subcommands.

    The group previously took the target as a positional on its own callback, which meant
    Click consumed the first token before it could ever match a subcommand: ``provenance
    list`` was silently reinterpreted as "trace the run called ``list``", failed to find
    it, and exited 1. Documented commands that cannot be reached are worse than absent
    ones, because the docs keep claiming they work.
    """

    def resolve_command(self, ctx: Any, args: list[str]) -> Any:
        first = next((a for a in args if not a.startswith("-")), None)
        if first is not None and first not in self.commands:
            args = ["trace", *args]
        return super().resolve_command(ctx, args)


app = typer.Typer(
    name="provenance",
    help="Walk the artifact provenance DAG.",
    no_args_is_help=True,
    cls=_TraceByDefault,
)
console = Console()


def _resolve_run_id(store: ArtifactStore, target: str) -> str:
    """Accept a run id, a run directory, or any file inside one."""
    path = Path(target)
    if path.exists():
        for parent in (path, *path.parents):
            record_path = parent / "run_record.json"
            if record_path.is_file():
                return str(json.loads(record_path.read_text(encoding="utf-8"))["run_id"])
        # A figure manifest names its sources rather than living in a run directory.
        if path.is_file() and path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            source = payload.get("source_run_id")
            if source:
                return str(source)
    return target


@app.command("trace")
def trace(
    target: Annotated[
        str,
        typer.Argument(help="Run id, run directory, artifact file, or figure_manifest.json"),
    ],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit a machine-readable summary")
    ] = False,
) -> None:
    """Print the lineage of an artifact, verifying every recorded hash on the way."""
    store = ArtifactStore()
    report = build_lineage(store, _resolve_run_id(store, target))

    if as_json:
        typer.echo(json.dumps(report.summary(), indent=2, sort_keys=True))
    else:
        typer.echo(render_lineage(report))

    if not report.complete:
        # A broken chain means a number cannot be traced to the bytes that produced it.
        raise typer.Exit(code=1)


@app.command("list")
def list_runs(
    stage: Annotated[str | None, typer.Option("--stage")] = None,
    experiment: Annotated[str | None, typer.Option("--experiment")] = None,
) -> None:
    """List finalized runs in the artifact tree."""
    from ocr_risk.schemas.enums import StageName

    store = ArtifactStore()
    stage_enum = StageName(stage) if stage else None
    records = store.find_runs(stage_enum, experiment)
    if not records:
        console.print("[yellow]no finalized runs found[/yellow]")
        return
    for record in records:
        flags = []
        if record.synthetic:
            flags.append("synthetic")
        if record.leaky:
            flags.append("LEAKY")
        if record.git.dirty:
            flags.append("dirty")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        console.print(
            f"{record.stage.value:<12} {record.run_id}  "
            f"{len(record.outputs)} outputs  {record.created_at_utc}{suffix}"
        )
