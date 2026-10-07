"""``ocr-risk config`` — inspect, validate, and export the configuration surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console

from ocr_risk.config import ExperimentConfig, load_config

app = typer.Typer(
    name="config", help="Inspect and validate experiment configuration.", no_args_is_help=True
)
console = Console()

OverrideOpt = Annotated[
    list[str] | None,
    typer.Option("--set", help="Override a value, e.g. --set risk.delta=0.05", show_default=False),
]


@app.command("schema")
def schema(
    output: Annotated[
        Path | None, typer.Option("--out", help="Write to a file instead of stdout")
    ] = None,
) -> None:
    """Emit the JSON Schema for an experiment configuration.

    Useful for editor validation, and as a machine-readable record of exactly which knobs
    exist at a given code version.
    """
    text = json.dumps(ExperimentConfig.model_json_schema(), indent=2, sort_keys=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        console.print(f"wrote {output}")
    else:
        typer.echo(text)


@app.command("show")
def show(
    config_path: Annotated[Path, typer.Argument(help="Path to an experiment config")],
    overrides: OverrideOpt = None,
) -> None:
    """Resolve a config and print the effective settings plus their hash.

    This is the same mapping that gets hashed into every run record, so it answers "what
    will actually run?" rather than "what did I type?".
    """
    resolved = load_config(config_path, overrides)
    typer.echo(yaml.safe_dump(resolved.mapping, sort_keys=True, allow_unicode=True))
    console.print(f"[bold]config_sha256[/bold] {resolved.sha256}")


@app.command("validate")
def validate(
    config_path: Annotated[Path, typer.Argument(help="Path to an experiment config")],
    overrides: OverrideOpt = None,
) -> None:
    """Validate a config without running anything."""
    resolved = load_config(config_path, overrides)
    cfg = resolved.config
    console.print(f"[green]valid[/green] {cfg.name}  sha256={resolved.sha256[:12]}")
    console.print(f"  datasets : {', '.join(cfg.enabled_dataset_ids) or '(none)'}")
    console.print(f"  engines  : {', '.join(cfg.enabled_engine_ids) or '(none)'}")
    console.print(f"  verifiers: {', '.join(v.id for v in cfg.verifiers) or '(none)'}")
    console.print(f"  protocol : {cfg.splits.protocol}")
    console.print(f"  epsilon  : {list(cfg.risk.epsilon_grid)}")
    if cfg.synthetic:
        console.print("  [yellow]SYNTHETIC — outputs are not research results[/yellow]")
    if cfg.splits.allow_document_overlap:
        console.print("  [red]LEAKY BY DESIGN — document overlap enabled (diagnostic only)[/red]")
